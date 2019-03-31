#include <bits/stdc++.h>
#include "operations.h"
using namespace std;


// Global Variables 
// n is input dimension, d is no of cell states
int n = 5, d = 3;	
vector< vector<double> > W_c, W_i, W_f, W_o, U_c, U_i, U_f, U_o, del_W;		
vector<double> a_, i_, f_, o_, a, i, f, o, C, h;
vector<double> gt, del_C, del_C_prev, del_h, del_a_, del_i_, del_f_, del_o_, del_a, del_i, del_f, del_o;


void LSTM_forward(vector<double> h_prev, vector<double> C_prev, vector<double> x)
{

	//forward pass for single LSTM cell
	a_ = add(mul(W_c, x), mul(U_c, h_prev));
	i_ = add(mul(W_i, x), mul(U_i, h_prev));
	f_ = add(mul(W_f, x), mul(U_f, h_prev));
	o_ = add(mul(W_o, x), mul(U_o, h_prev));

	
	a = tanh_vec(a_);
	i = sigmoid_vec(i_);
	f = sigmoid_vec(f_);
	o = sigmoid_vec(o_);
	
	C = add(dot(i, a), dot(f, C_prev));
	h = dot(o, tanh_vec(C));

}


void LSTM_backward(vector<double> h_prev, vector<double> C_prev, vector<double> x)
{
	// backward pass for single LSTM cell
	
	// exact del_h depends on the loss function, assuming cross-entropy loss
	for(int j=0; j<d; j++)
		del_h.push_back(-gt[j]/h[j]);
	del_o = dot(del_h, tanh_vec(C));
	del_C = dot(dot(o, tanh_derv(C)), del_h);
	del_i = dot(del_C, a);
	del_f = dot(del_C, C_prev);
	del_a = dot(del_C, i);
	del_C_prev = dot(del_C, f);

	del_a_ = dot(del_a, tanh_derv(a_));
	del_i_ = dot(del_i, sigm_derv(i));
	del_f_ = dot(del_f, sigm_derv(f));
	del_o_ = dot(del_o, sigm_derv(o));

	//concatenate del_a_, del_i_, del_f_, del_o_
	del_a_.insert(del_a_.end(), del_i_.begin(), del_i_.end());
	del_a_.insert(del_a_.end(), del_f_.begin(), del_f_.end());
	del_a_.insert(del_a_.end(), del_o_.begin(), del_o_.end());
	
	//concatenate x, h_prev
	x.insert(x.end(), h_prev.begin(), h_prev.end()); 

	del_W = mul_vec(del_a_, x);

	// dimension of gradient matrix is 4d X (n+d), i.e. 12 X 8
	cout<<"Weight gradient matrix: "<<endl;
	print_wt(del_W);
}


int main()
{
	vector<double> h_prev, x, C_prev;
	

	//initializing vectors
	rand_init_wt(W_c, d, n); rand_init_wt(U_c, d, d);
	rand_init_wt(W_o, d, n); rand_init_wt(U_o, d, d);
	rand_init_wt(W_f, d, n); rand_init_wt(U_f, d, d);
	rand_init_wt(W_i, d, n); rand_init_wt(U_i, d, d);
	rand_init_vec(h_prev, d); 
	rand_init_vec(C_prev, d);
	rand_init_vec(x, n);


	// setting a ground truth value
	for(int j=0; j<d; j++)
		gt.push_back(0);
	gt[d-1] = 1;

	LSTM_forward(h_prev,C_prev,x);
	LSTM_backward(h_prev,C_prev,x);

	// numerical gradient check for i,j entry of a weight matrix
	int i=2,j =1;
	
	LSTM_forward(h_prev,C_prev,x);
	double loss_prev = -log(h[d-1]);
	
	double delta = 0.1;
	W_c[i][j] += delta;
	
	LSTM_forward(h_prev,C_prev,x);
	double loss_new = -log(h[d-1]);

	cout<<loss_new-loss_prev<<endl;

}