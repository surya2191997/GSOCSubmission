#include <bits/stdc++.h>
using namespace std;


vector<double> mul(vector< vector<double> > A, vector<double> B)
{

	vector<double> ans;
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

vector< vector<double> > mul_vec(vector<double> A, vector<double> B)
{
	vector< vector<double> > ans;
	vector<double> v;

	for(int i=0; i<A.size(); i++)
	{
		for(int j=0; j<B.size(); j++)
			v.push_back(A[i]*B[j]);
		ans.push_back(v);
		v.clear();
	}

	return ans;
} 


vector<double> add(vector<double> A, vector<double> B)
{
	for(int i=0; i<A.size(); i++)
		A[i] = A[i] + B[i];
	return A;
} 

vector<double> dot(vector<double> A, vector<double> B)
{
	for(int i=0; i<A.size(); i++)
		A[i] = A[i]*B[i];
	return A;
} 


vector<double> sigmoid_vec(vector<double> A)
{
	for(int i=0; i<A.size(); i++)
		A[i] = 1/(1+exp(-A[i]));
	return A;
}

vector<double> tanh_vec(vector<double> A)
{
	for(int i=0; i<A.size(); i++)
		A[i] = tanh(A[i]);
	return A;
}


vector<double> tanh_derv(vector<double> A)
{
	for(int i=0; i<A.size(); i++)
		A[i] = 1 - tanh(A[i])*tanh(A[i]);
	return A;
}


vector<double> sigm_derv(vector<double> A)
{
	for(int i=0; i<A.size(); i++)
		A[i] = A[i]*(1-A[i]);
	return A;
}

void rand_init_wt(vector< vector<double> > &W, int x, int y)
{
	vector<double> v;
	for(int i=0; i<x; i++)
	{
		for(int j=0;j<y;j++)
			v.push_back(rand()*1.0/RAND_MAX);
		W.push_back(v);
		v.clear();
	}
}

void rand_init_vec(vector<double> &V, int n)
{
	for(int i=0; i<n; i++)
		V.push_back(rand()*1.0/RAND_MAX);
}


void print_wt(vector< vector<double> > &W)
{
	for(int j=0;j<W.size();j++)
	{
		for(int k=0;k<W[0].size();k++)
			cout<<W[j][k]<<" ";
		cout<<endl;
	}
}

