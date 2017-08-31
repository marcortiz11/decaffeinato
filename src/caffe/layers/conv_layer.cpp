#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  in = 0;
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  seed_val = time(NULL);
  seed = seed_val;

  //Debugger per weight updates
  /*if(this->layer_param_.debug() && this->layer_param_.debug_params().weight_updates() && in >= this->layer_param_.debug_params().epoch_start() && in <= this->layer_param_.debug_params().epoch_end()){
	  cout << "START_WEIGHT_UPDATES" << endl << endl;
	  if(this->layer_param_.debug_params().type() == 2){
		/*int offset = this->layer_param_.offset();
		  int layer_id = this->layer_param_.layer_id();
		  int neuron = 0;
		  for( int weight = 0; weight < this->blobs_[0]->count();++weight){
			Dtype before = (this->blobs_[0]->mutable_cpu_data())[weight];
			  if(this->layer_param_.floating_point()){
				RoundFloatingPoint(this->layer_param_, this->blobs_[0]->mutable_cpu_data(),1);
			  }else if(fixedPrecision_){
				RoundFixedPoint(this->layer_param_ ,this->blobs_[0]->mutable_cpu_data(),1, &seed);
			  }
			Dtype after = (this->blobs_[0]->mutable_cpu_data())[weight];
		   	cout << "2:" << offset + neuron << ":1:"<< layer_id << ":" <<  neuron << ":" << weight;
			cout << in << ":90000000:" << after-before << endl;
			neuron += weight%(this->blobs_[0]->count()/top[0]->count());
		  }
	 }else{
		//Imprimim tots els updates de cop
		for(int weight = 0; weight<this->blobs_[0]->count(); ++weight){
			Dtype before = (this->blobs_[0]->mutable_cpu_data())[weight];
			if(this->layer_param_.floating_point()){
				RoundFloatingPoint(this->layer_param_, this->blobs_[0]->mutable_cpu_data(),1);
			  }else if(this->layer_param_.fixed_precision()){
				RoundFixedPoint(this->layer_param_ ,this->blobs_[0]->mutable_cpu_data(),1, &seed);
			  }
			Dtype after = (this->blobs_[0]->mutable_cpu_data())[weight];
			cout << in << ":" << after-before << endl;
		}	 	
	}
	cout << "END_WEIGHT_UPDATES" << endl << endl;
  }*/
/*
  if(this->layer_param_.floating_point()){
        RoundFloatingPoint(this->layer_param_, this->blobs_[0]->mutable_cpu_data(),this->blobs_[0]->count());
  }else if(this->layer_param_.fixed_precision()){
        RoundFixedPoint(this->layer_param_ ,this->blobs_[0]->mutable_cpu_data(),this->blobs_[0]->count(), &seed);
  }

  if(this->bias_term_){
     	seed = seed_val;
	if(this->layer_param_.floating_point()){
                RoundFloatingPoint(this->layer_param_, this->blobs_[1]->mutable_cpu_data(),this->blobs_[1]->count());
	}else if(this->layer_param_.fixed_precision()){
		RoundFixedPoint(this->layer_param_ ,this->blobs_[1]->mutable_cpu_data(),this->blobs_[1]->count(), &seed);
	}
  }*/

   //Es fa debug dels pesos
  /*if(this->layer_param_.debug() && this->layer_param_.debug_params().weights() && in >= this->layer_param_.debug_params().epoch_start() && in <= this->layer_param_.debug_params().epoch_end()){
	  cout << "START_WEIGHTS" << endl << endl;
	  if(this->layer_param_.debug_params().type() == 2){
		  /*int offset = this->layer_param_.offset();
		  int layer_id = this->layer_param_.layer_id();
		  int neuron = 0;
		  for( int weight = 0; weight < this->layer_param_.inner_product_param().num_output();weight++){
		   	cout << "2:" << offset + neuron << ":1:"<< layer_id << ":" <<  neuron << ":" << weight;
			cout << in << ":90000000:" << (this->blobs_[0]->cpu_data())[weight] << endl;
			neuron += weight%(this->blobs_[0]->count()/top[0]->count());
		  }
		  //Entrada
	 }else{
		//Imprimim tots els pesos de cop
		for(int i = 0; i<this->blobs_[0]->count(); ++i) cout << in << ":" << (this->blobs_[0]->cpu_data())[i] << endl; 	
	}
        cout << "END_WEIGHTS" << endl << endl;
  }*/

  const Dtype* weight = this->blobs_[0]->cpu_data();

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
  //Arrodonir part convolutional


  //Es fa debug de les sortides
  /*if(this->layer_param_.debug() && this->layer_param_.debug_params().outputs() && in >= this->layer_param_.debug_params().epoch_start() && in <= this->layer_param_.debug_params().epoch_end()){
	  cout << "START_OUTPUTS" << endl << endl;
	  if(this->layer_param_.debug_params().type() == 2){
		  int offset = this->layer_param_.offset();
		  int layer_id = this->layer_param_.layer_id();
		  for( int neuron = 0; neuron < this->layer_param_.inner_product_param().num_output(); ++neuron){
		   	cout << "2:" << offset + neuron << ":1:"<< layer_id << ":" <<  neuron << ":";
			cout << in << ":90000000:" << (top[0]->cpu_data())[neuron] << endl;
		  }
		  //Entrada
	 }else{
		//Imprimim totes les sortides de cop
		for(int i = 0; i<top[0]->count(); ++i) cout << in << ":" << (top[0]->cpu_data())[i] << endl; 	
	}
	cout << "END_OUTPUTS" << endl << endl;
  }*/


}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
 
  seed = seed_val;

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    //Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }

   seed = seed_val;
   //Arrodoniment
   /*if(this->layer_param_.fixed_precision()) RoundFixedPoint(this->layer_param_, bottom[0]->mutable_cpu_diff(), top.size()*bottom[0]->count(), &seed);
   else if(this->layer_param_.floating_point()){
	 RoundFloatingPoint(this->layer_param_,bottom[0]->mutable_cpu_diff(), top.size()*bottom[0]->count());
   }*/

   //Es fa debug dels errors propagats
    /*if(this->layer_param_.debug() && this->layer_param_.debug_params().error() && in >= this->layer_param_.debug_params().epoch_start() && in <= this->layer_param_.debug_params().epoch_end()){
	  cout << "START_ERROR" << endl << endl;
	  if(this->layer_param_.debug_params().type() == 2){
		  /*int offset = this->layer_param_.offset();
		  int layer_id = this->layer_param_.layer_id();
		  for( int neuron = 0; neuron < this->layer_param_.inner_product_param().num_output(); ++neuron){
		   	cout << "2:" << offset + neuron << ":1:"<< layer_id << ":" <<  neuron << ":";
			cout << in << ":90000000:" << (top[0]->cpu_data())[neuron] << endl;
		  }
		  //Entrada
	 }else{
		//Imprimim tots els errors propagats
		for(int i = 0; i<bottom[0]->count(); ++i) cout << in << ":" << (bottom[0]->cpu_diff())[i] << endl; 	
	}
	cout << "END_ERROR" << endl << endl;
   }*/

    /*Dtype *r = bottom[0]->mutable_cpu_diff();
    int elems = top.size()*bottom[0]->count();

    int EXP = 4;
    int MANT = 0;
    //string type = ";
    int zero =  15;

    if (zero == -1) zero = pow(2,EXP-1);
    if (EXP < 0 || MANT < 0 ) return;
    if (EXP > 11|| MANT > 53) cout << "WARNING: Possible loss of precision" << endl;
	
    double MAX = pow(2,pow(2, EXP)-1-zero) * (2-pow(2,-MANT));
    double smallest = pow(2,-MANT);

    Dtype rfp;
    for(int i = 0; i<elems; ++i){
        if(r[i] > MAX) r[i] = MAX;
        else if (r[i] < -MAX) r[i] = -MAX;
        else{
                int exp; frexp(r[i],&exp);exp -= 1;
                double epsilon = pow(2,exp) * smallest;
                
                //Limit the exponent by smaller value
                if(exp < (-zero))  {
                    exp = -zero;
                    epsilon = pow(2,exp);
                }
                
                //Stochastic Rounding Algorithm
                long int multipleEpsilon = (long int) (r[i]/epsilon);
                rfp = (double) multipleEpsilon * epsilon;
                float numberToBeat = (rand() % 100)/100.0;
                r[i] = ((double) abs((r[i]-rfp)) / epsilon) > numberToBeat ? ( -(r[i] < 0) * epsilon + (r[i] > 0) * epsilon + rfp) : rfp;
        }
    }*/

  /*if (this->bias_term_ && this->param_propagate_down_[1]) {
      seed = seed_val;
      if(this->layer_param_.fixed_precision()) RoundFixedPoint(this->layer_param_, this->blobs_[1]->mutable_cpu_diff(), this->blobs_[1]->count(),&seed);
      else if(this->layer_param_.floating_point()){
		        RoundFloatingPoint(this->layer_param_,this->blobs_[1]->mutable_cpu_diff(),this->blobs_[1]->count());
      }
  }*/
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
