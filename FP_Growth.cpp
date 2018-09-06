#include<iostream>
#include<string>
#include<vector>
#include<ifsream>
using namespace std;

struct FP_Node
{
	int fre;
	FP_Node * sub;
	FP_Node * S_N;
};

struct Header 
{
	int counter;
	string s;	//Record the string 
	FP_Node * p;
};

class FP_Tree
{
	FP_Tree();
	~FP_Tree();
	FP_Tree(vector<string>,int);		
	FP_Tree
};

void Find_MostFreq_SingleItem(string filename, vector<string> & s, vector<int> & f)
{
//Search file and calculate each single feature's apperance,and sort them into two vectors
	ifstream fread(filename);
	string feature;
	while(fin>>feature)
	{
		int i = Find_String(s,feature);
		if(i != -1)
		{
			f[i]++;		
		}	
		else 
		{
			s.push_back(feature);
			f.push_back(1);
		}
	}
	fread.close();
	//sort in DESC
		
}














