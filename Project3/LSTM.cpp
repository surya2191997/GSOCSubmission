#include <bits/stdc++.h>
using namespace std;

vector<float> mul(vector< vector<float> > A, vector<float> B)
{

	vector<float> ans;
	int r = A.size(), c = A[0].size(), sum = 0;
	for(int i = 0; i < r; i++)
	{
		for(int j = 0; j < c; j++)
			sum += A[i][j]*B[j];
		ans.push_back(sum);
		sum = 0;
	}

	return ans;
} 

vector< vector<float> > mul_vec(vector<float> A, vector<float> B)
{
	vector< vector<float> > ans;
	vector<float> v;

	for(int i=0; i<A.size(); i++)
	{
		for(int j=0; j<B.size(); j++)
			v.push_back(A[i]*B[j]);
		ans.push_back(v);
		v.clear();
	}

	return ans;
} 


vector<float> add(vector<float> A, vector<float> B)
{
	for(int i=0; i<A.size(); i++)
		A[i] = A[i] + B[i];
	return A;
} 

vector<float> dot(vector<float> A, vector<float> B)
{
	for(int i=0; i<A.size(); i++)
		A[i] = A[i]*B[i];
	return A;
} 


vector<float> sigmoid_vec(vector<float> A)
{
	for(int i=0; i<A.size(); i++)
		A[i] = 1/(1+exp(-A[i]));
	return A;
}

vector<float> tanh_vec(vector<float> A)
{
	for(int i=0; i<A.size(); i++)
		A[i] = tanh(A[i]);
	return A;
}


vector<float> tanh_derv(vector<float> A)
{
	for(int i=0; i<A.size(); i++)
		A[i] = 1 - tanh(A[i])*tanh(A[i]);
	return A;
}


vector<float> sigm_derv(vector<float> A)
{
	for(int i=0; i<A.size(); i++)
		A[i] = A[i]*(1-A[i]);
	return A;
}

//LSTM parameters
// n is dimension of the input vector
// d is the dimension of the cell state
int n = 5, d = 3;								
vector< vector<float> > W_c, W_i, W_f, W_o;		
vector< vector<float> > U_c, U_i, U_f, U_o;



void LSTM(vector<float> h_prev, vector<float> C_prev, vector<float> x)
{

	//forward pass for single LSTM cell


	vector<float> a_ = add(mul(W_c, x), mul(U_c, h_prev));
	vector<float> i_ = add(mul(W_i, x), mul(U_i, h_prev));
	vector<float> f_ = add(mul(W_f, x), mul(U_f, h_prev));
	vector<float> o_ = add(mul(W_o, x), mul(U_o, h_prev));

	
	vector<float> a = tanh_vec(a_);
	vector<float> i = sigmoid_vec(i_);
	vector<float> f = sigmoid_vec(f_);
	vector<float> o = sigmoid_vec(o_);
	
	vector<float> C = add(dot(i, a), dot(f, C_prev));
	vector<float> h = dot(o, tanh_vec(C));



	// backward pass for single LSTM cell

	// exact del_h depends on the loss function, assuming cross-entropy loss
	vector<float> del_h, gt; 
	
	// setting a ground truth value
	for(int j=0; j<d; j++)
		gt.push_back(0);
	gt[d-1] = 1;

	for(int j=0; j<d; j++)
		del_h.push_back(-gt[j]/h[j]);
	vector<float> del_o = dot(del_h, tanh_vec(C));
	vector<float> del_C = dot(dot(o, tanh_derv(C)), del_h);
	vector<float> del_i = dot(del_C, a);
	vector<float> del_f = dot(del_C, C_prev);
	vector<float> del_a = dot(del_C, i);
	vector<float> del_C_prev = dot(del_C, f);

	vector<float> del_a_ = dot(del_a, tanh_derv(a_));
	vector<float> del_i_ = dot(del_i, sigm_derv(i));
	vector<float> del_f_ = dot(del_f, sigm_derv(f));
	vector<float> del_o_ = dot(del_o, sigm_derv(o));

	//concatenate del_a_, del_i_, del_f_, del_o_
	del_a_.insert(del_a_.end(), del_i_.begin(), del_i_.end());
	del_a_.insert(del_a_.end(), del_f_.begin(), del_f_.end());
	del_a_.insert(del_a_.end(), del_o_.begin(), del_o_.end());
	
	//concatenate x, h_prev
	x.insert(x.end(), h_prev.begin(), h_prev.end()); 

	vector< vector<float> > del_W = mul_vec(del_a_, x);


	for(int j=0;j<del_W.size();j++)
	{
		for(int k=0;k<del_W[0].size();k++)
			cout<<del_W[j][k]<<" ";
		cout<<endl;
	}
		

}



int main()
{
	vector<float> v, h, x, C;
	

	//initializing vectors
	for(int i=0; i<n; i++)
	{
		for(int j=0;j<d;j++)
			v.push_back(rand()*1.0/RAND_MAX);
		W_c.push_back(v);
		W_o.push_back(v);
		W_f.push_back(v);
		W_i.push_back(v);
		v.clear();		
	}

	for(int i=0; i<d; i++)
	{
		for(int j=0;j<d;j++)
			v.push_back(rand()*1.0/RAND_MAX);
		U_c.push_back(v);
		U_o.push_back(v);
		U_f.push_back(v);
		U_i.push_back(v);
		v.clear();		
	}

	for(int i=0; i<d; i++)
	{
		h.push_back(rand()*1.0/RAND_MAX);
		C.push_back(rand()*1.0/RAND_MAX);
	}

	for(int i=0; i<n; i++)
		x.push_back(rand()*1.0/RAND_MAX);

	LSTM(h,C,x);
}