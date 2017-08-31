#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include <iostream>
using namespace std;

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}




void stochasticRounding(double MAX, double MIN, double epsilon, double* r, int elems){
    double rfp;
    for(int i = 0; i<elems; ++i){
        if(r[i] > MAX) r[i] = MAX;
        else if (r[i] < MIN) r[i] = MIN;
        else{
                long int multipleEpsilon = (r[i]/epsilon);
                rfp = (double) multipleEpsilon * epsilon;
                float numberToBeat = (rand() % 100)/100.0;
                r[i] = ((double) abs((r[i]-rfp)) / epsilon) > numberToBeat ? ( -(r[i] < 0) * epsilon + (r[i] > 0) * epsilon + rfp) : rfp;
        }
    }
}


void stochasticRounding(double MAX, double smallest, double *r, int elems, int zero){
    double rfp;
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
    }
}


void roundToNearest(double MAX, double MIN, double epsilon, double* r, int elems){
    double rfp;
    for(int i = 0; i<elems; ++i){
        if(r[i] > MAX) r[i] = MAX;
        else if (r[i] < MIN) r[i] = MIN;
        else{
                long int multipleEpsilon = (int) (r[i]/epsilon);
                rfp = (double) multipleEpsilon * epsilon;
                r[i] = abs(r[i]-rfp) >= (epsilon/2) ? ( -(r[i] < 0) * epsilon + (r[i] > 0) * epsilon + rfp) : rfp;
        }
    }
}



void roundToNearest(double MAX, double smallest, double* r, int elems, int zero){
    double rfp;
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
                r[i] = ( abs(r[i]-rfp) > epsilon/2 ) ? ( -(r[i] < 0) * epsilon + (r[i] > 0) * epsilon + rfp) : rfp;
        }
    }
}



void roundWithProb(double MAX, double smallest, double *r, int elems, float PROB){
    double rfp;
    for(int i = 0; i<elems; ++i){
        if(r[i] >= MAX) r[i] = MAX;
        else if (r[i] <= -MAX) r[i] = -MAX;
        else{
                int exp; frexp(r[i],&exp);
                float epsilon = pow(2,exp) * smallest;
                long int multipleEpsilon = (long int) (r[i]/epsilon);
                rfp = (double) multipleEpsilon * epsilon;
                double numberToBeat = (rand() % 100)/100.0;
                r[i] = rfp + (PROB > numberToBeat ? -(rfp<0)*epsilon + (rfp>0)*epsilon : 0);
        }
    }
}


void roundWithProb(double MAX, double MIN, double epsilon, double *r, int elems, float PROB){
    double rfp;
    for(int i = 0; i<elems; ++i){
        if(r[i] > MAX) r[i] = MAX;
        else if (r[i] < MIN) r[i] = MIN;
        else{
                long int multipleEpsilon = (int) (r[i]/epsilon);
                rfp = (float) multipleEpsilon * epsilon;
                double numberToBeat = (rand() % 100)/100.0;
                r[i] = PROB > numberToBeat ? ( -(r[i] < 0) * epsilon + (r[i] > 0) * epsilon + rfp) : rfp;
        }
    }
}



//Arrodoniment Stochastic Rounding + Round to nearest amb probabilitat
void roundMixed(const double MAX, const double MIN, const double epsilon, double* r, int elems, const float PROB, unsigned int *seed, const int step, const int block){
    float rfp;
    bool first_rounding = true;
    for(int i = 0; i<elems; ++i){
	if(i%(step * block) == 0) {
		float num = (rand_r(seed) % 100) / 100.0;
		first_rounding = PROB > num;
	}
        if(r[i] > MAX) r[i] = MAX;
        else if (r[i] < MIN) r[i] = MIN;
        else{
		int multipleEpsilon = (r[i]/epsilon);
                rfp = (float) multipleEpsilon * epsilon;
		if(first_rounding){
			float numberToBeat = (rand() % 100)/100.0;
			r[i] = ((double) abs((r[i]-rfp)) / epsilon) > numberToBeat ? ( -(r[i] < 0) * epsilon + (r[i] > 0) * epsilon + rfp) : rfp;
		}else {
		        r[i] = abs(r[i]-rfp) >= (epsilon/2) ? ( -(r[i] < 0) * epsilon + (r[i] > 0) * epsilon + rfp) : rfp;
		}        
        }
    }
}	


void Truncate(const double MAX, const double MIN, const double epsilon, double* r, int elems){
    for(int i = 0; i<elems; ++i){
        if(r[i] > MAX) r[i] = MAX;
        else if (r[i] < MIN) r[i] = MIN;
        else{
		long int multipleEpsilon = (r[i]/epsilon);
                r[i] = (double) multipleEpsilon * epsilon;
		     
        }
    }
}


void Truncate(double MAX, double smallest, double *r, int elems, int zero){
   // double rfp;
    for(int i = 0; i<elems; ++i){
        if(r[i] > MAX) r[i] = MAX;
        else if (r[i] < -MAX) r[i] = -MAX;
        else{
                int exp; frexp(r[i],&exp);exp -= 1;
                double epsilon = pow(2,exp) * smallest;
                
                if(exp < (-zero))  {
                    exp = -zero;
                    epsilon = pow(2,exp);
                }

                long int multipleEpsilon = (long int) (r[i]/epsilon);
                r[i] = (double) multipleEpsilon * epsilon;
        }
    }
}


template <>
void RoundFixedPoint<double>(const caffe::CustomDataType& layer_param, double* p, int elems){

    //Param extraction
    const int INTEGER = layer_param.enter();
    const int FRACT = layer_param.fraccio();
    const string type = layer_param.rounding();
    const float PROB = layer_param.prob();

    if(INTEGER < 0 || FRACT < 0) return;

    double MAX,MIN,epsilon;
    epsilon = pow(2, -FRACT);
    MAX = pow(2,INTEGER)-epsilon;
    MIN = -pow(2,INTEGER);   

    if (type == "stochastic") stochasticRounding(MAX,MIN,epsilon,p, elems);
    else if (type == "nearest") roundToNearest(MAX,MIN,epsilon,p,elems);
    else if (type == "withProb") roundWithProb(MAX,MIN,epsilon, p, elems, PROB); 
    else if (type == "truncate") Truncate(MAX,MIN,epsilon,p,elems);
    else if (type == "mixed") {
		/*int step; 
		if(layer_param.type() == "Convolution") step = elems / layer_param.convolution_param().num_output(); 
		else step = elems / layer_param.inner_product_param().num_output(); 
		const int block = layer_param.precision().block();
		roundMixed(MAX ,MIN, epsilon, p, elems, PROB, seed, step, block);*/
    }
}

template<>
void RoundFloatingPoint<double>(const caffe::CustomDataType& layer_param, double *p, const int elems){

    int EXP = layer_param.exp();
    int MANT = layer_param.mant();
    //float PROB = layer_param.fpoint().prob();
    string type = layer_param.rounding();
    int zero =  layer_param.zero();

    if (zero == -1) zero = pow(2,EXP-1);
    if (EXP < 0 || MANT < 0 ) return;
	
    double MAX = pow(2,pow(2, EXP)-1-zero) * (2-pow(2,-MANT));
    double smallest = pow(2,-MANT);

    if (type == "stochastic") stochasticRounding(MAX,smallest,p,elems,zero);
    else if (type == "nearest") roundToNearest(MAX,smallest,p,elems, zero);
    else if (type == "truncate") Truncate(MAX,smallest,p,elems,zero);
    //else if (type == "withProb") roundWithProb(MAX,smallest,p,elems, PROB);

}

template <>
void Round<double>(const caffe::CustomDataType& dataType, double* p, int elems){
	if(dataType.fixed_point()) RoundFixedPoint(dataType,p,elems);
	else RoundFloatingPoint(dataType,p,elems);
}	

template <>
void Round<float>(const caffe::CustomDataType& dataType, float* p, int elems){
}





template <>
void RoundFixedPoint<float>(const caffe::CustomDataType& layer_param, float* p, int elems){
    //NOT IMPLEMENTED
}

template <>
void RoundFloatingPoint<float>(const caffe::CustomDataType& layer_param, float *p, const int elems){
    //NOT IMPLEMENTED
}
    
}  // namespace caffe
