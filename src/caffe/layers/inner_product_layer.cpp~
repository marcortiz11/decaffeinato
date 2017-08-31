#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include <fstream>

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  
  /*fixedPrecision_ = this->layer_param_.fixed_precision();
  INT_ = this->layer_param_.precision().enter();
  FRACT_ = this->layer_param_.precision().fraccio();
  type_ = this->layer_param_.precision().rounding();*/
  in = 0;
  
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    
    
    
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
    
    
    //No arrodonim les bias ja que el seu valor és 0.
    
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //Random sequence for mixed_rounding
  seed_val = time(NULL);
  seed = seed_val;
  ++in;
  //Debugger per weight updates
  /*if(this->layer_param_.debug() && this->layer_param_.debug_params().weight_updates() && in >= this->layer_param_.debug_params().epoch_start() && in <= this->layer_param_.debug_params().epoch_end()){
          cout << "START_WEIGHT_UPDATES" << endl << endl;
	  if(this->layer_param_.debug_params().type() == 2){
		  int offset = this->layer_param_.offset();
		  int layer_id = this->layer_param_.layer_id();
		  int neuron = 0;
		  for( int weight = 0; weight < this->blobs_[0]->count();++weight){
			Dtype before = (this->blobs_[0]->mutable_cpu_data())[weight];
			  if(this->layer_param_.floating_point()){
				RoundFloatingPoint(this->layer_param_, this->blobs_[0]->mutable_cpu_data() + weight,1);
			  }else if(fixedPrecision_){
				RoundFixedPoint(this->layer_param_ ,this->blobs_[0]->mutable_cpu_data() + weight,1, &seed);
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
			  }else if(fixedPrecision_){
				RoundFixedPoint(this->layer_param_ ,this->blobs_[0]->mutable_cpu_data(),1, &seed);
			  }
			Dtype after = (this->blobs_[0]->mutable_cpu_data())[weight];
			cout << in << ":" << after-before << endl;
		}	 	
	}
 	cout << "END_WEIGHT_UPDATES" << endl << endl;
  }*/

  /*if(this->layer_param_.floating_point()){
	RoundFloatingPoint(this->layer_param_, this->blobs_[0]->mutable_cpu_data(),this->blobs_[0]->count());
  }else if(fixedPrecision_){
	RoundFixedPoint(this->layer_param_ ,this->blobs_[0]->mutable_cpu_data(),this->blobs_[0]->count(), &seed);
  }*/

  //Es fa debug dels pesos
  /*if(this->layer_param_.debug() && this->layer_param_.debug_params().weights() && in >= this->layer_param_.debug_params().epoch_start() && in <= this->layer_param_.debug_params().epoch_end()){
 	  cout << "START_WEIGHTS" << endl << endl;
	  if(this->layer_param_.debug_params().type() == 2){
		  int offset = this->layer_param_.offset();
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

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);

  if (bias_term_) {
  	/*seed = seed_val;
       if(this->layer_param_.floating_point()){
                RoundFloatingPoint(this->layer_param_,this->blobs_[1]->mutable_cpu_data(),this->blobs_[1]->count());
        }else if(fixedPrecision_){
        	RoundFixedPoint(this->layer_param_ ,this->blobs_[1]->mutable_cpu_data(),this->blobs_[1]->count(), &seed);
  	}*/
      
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }

}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) { 

  //Start again the sequence
  seed = seed_val;

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());

      /*seed = seed_val;
	//ARRODONIM
      if(fixedPrecision_) RoundFixedPoint(this->layer_param_, this->blobs_[1]->mutable_cpu_diff(), this->blobs_[1]->count(),&seed);
      else if(this->layer_param_.floating_point()){
		        RoundFloatingPoint(this->layer_param_,this->blobs_[1]->mutable_cpu_diff(),this->blobs_[1]->count());
      }*/

  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
    
    //Arrodoniment
    /*if(fixedPrecision_) RoundFixedPoint(this->layer_param_, bottom[0]->mutable_cpu_diff(), bottom[0]->count(),&seed);
    else if(this->layer_param_.floating_point()){
                RoundFloatingPoint(this->layer_param_,bottom[0]->mutable_cpu_diff(),bottom[0]->count());
    }*/

    //Es fa debug dels errors propagats

    /*if(this->layer_param_.debug() && this->layer_param_.debug_params().error() && in >= this->layer_param_.debug_params().epoch_start() && in <= this->layer_param_.debug_params().epoch_end()){
	  cout << "START_ERROR" << endl << endl;
	  if(this->layer_param_.debug_params().type() == 2){
		  int offset = this->layer_param_.offset();
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
    fo r(int i = 0; i<elems; ++i){
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

  }
  
  
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
