#include<iostream>
#include<string>
#include<vector>
#include<ifsream>
using namespace std;

struct FP_Node
{
	string context;
	int fre;
	FP_Node * sub;
	FP_Node * S_N;
};

struct Header 
{
	int counter;
	string s;	//Record the string 
	FP_Node * p;
	Header * next;
};

class Header_Linklist
{
public:
	Header_Linklist();
	~Header_Linklist();
	void Insert(string,int);
private:
	int length;
	Header * head;
}

class FP_Tree
{
	FP_Tree();
	~FP_Tree();
	FP_Tree(vector<string>,int);		
	void Insert()
};

int Find_String(vector<string> s, string p)
{
	for(int i = 0; i < s.size(); i++)
	{
		if(s[i] == p) return i;
	}
	return -1;
}

Header_Linklist::Header_Linklist()
{
	head = new Header;
	head->counter = 0;
	length = 0;
	head->P = NULL;
	head->s = NULL;
	head->next = NULL;
}

Header_Linklist::Insert(string s, int i)
{
	Header * temp;
	temp = new Header;
	temp->counter = i;
	temp->s = s;
	temp->next = NULL;
	Header * tail = new Header;
	tail = head;
	while(tail->next)
	{
		tail = tail->next;
	}
	tail->next = temp;
}

void quicksort(vector<int> &f, vector<string> & s, int p, int q)
{
	int i = p;
	int j = q;
	int temp = f[p];
	int temp_index = p;
	int temp_string = s[p];
	//switch small to left and big to right
	while(i < j)
	{
		while(f[j] <= temp && j > i)	j--;
		if(j > i)
		{
			f[i] = f[j];
			s[i] = s[j];
			i++;

			while(f[i] > temp && i < j) i++;
			if(i < j)
			{
				f[i] = f[j];
				s[i] = s[j];
				j--;
			}
		}
	}
	f[i] = temp;
	s[i] = temp_string;

	if(p < (i-1)) quicksort(f,s,p,j-1);
	if((j+1) < q) quicksort(f,s,j+1,q);
}

void Find_MostFreq_SingleItem(string filename, vector<string> & s, vector<int> & f,Header_Linklist * hl)
{
//Search file and calculate each single feature's apperance,and sort them into two vectors
	ifstream fread(filename);
	string feature;
	vector<string> content,data;
	string d = ",";
	while(fin>>content)
	{
		data.push_back(content);
		//拆分content(,)然后得到feature(单个)
		for(int n = 0; n< content.length(); n++)
		{
			if(content[n] != d)
			{	
				feature = content[n];
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
			else continue;
		}
	}
	fread.close();
	//sort in DESC
	quicksort(f,s,0,f.size()-1);

	//将s与f写入hl中
	for(int index = 0 ; index < s.size(); index++)
	{
		hl->Insert(s[index],f[index]);
	}

	//将降序的f与s插入到root下
	for(int i = 0; i < data.size(); i++)
	{
		string cur_data = data[i];
		
	}

}
