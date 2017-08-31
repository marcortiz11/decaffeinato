#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ++in;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
  /*bool fixedPrecision_ = this -> layer_param_.fixed_precision();

  if(fixedPrecision_) RoundFixedPoint(this->layer_param_, top[0]->mutable_cpu_data(), top[0]->count(), NULL);
  else if(this->layer_param_.floating_point()){
               RoundFloatingPoint(this->layer_param_,top[0]->mutable_cpu_data(), top[0]->count());
  }
    //Es fa debug de les sortides
  if(this->layer_param_.debug() && this->layer_param_.debug_params().outputs() && in >= this->layer_param_.debug_params().epoch_start() && in <= this->layer_param_.debug_params().epoch_end()){
	  cout << "START_OUTPUTS" << endl << endl;
	  if(this->layer_param_.debug_params().type() == 2){
		  int offset = this->layer_param_.offset();
		  int layer_id = this->layer_param_.layer_id();
		  for( int neuron = 0; neuron < this->layer_param_.inner_product_param().num_output(); ++neuron){
		   	cout << "2:" << offset + neuron << ":1:"<< layer_id << ":" <<  neuron << ":";
			cout << in << ":90000000:" << (top[0]->cpu_data())[neuron] << endl;
		  }
		  //Entrada
	 }else if(this->layer_param_.debug_params().type()==1){
		//Imprimim totes les sortides de cop
		for(int i = 0; i<top[0]->count(); ++i) cout << in << ":" << (top[0]->cpu_data())[i] << endl; 	
	}
	cout << "END_OUTPUTS" << endl << endl;
  }*/
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
  
  if (propagate_down[0]) {

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }

  }
  //No need to round the gradient 
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
