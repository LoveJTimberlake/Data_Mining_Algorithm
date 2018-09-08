#include<iostream>
#include<string>
#include<vector>
#include<ifsream>
using namespace std;

struct FP_Node
{
    string context;
    int fre;
    vector<FP_Node *> sub_list;
    FP_Node * S_N;
};

struct Header
{
    int counter;
    string s;    //Record the string
    FP_Node * p;
    Header * next;
};

class Header_Linklist
{
public:
    Header_Linklist();
    ~Header_Linklist();
    void Insert(string,int);
    bool Match(string);
private:
    int length;
    Header * head;
}

class FP_Tree
{
public:
    FP_Tree();
    ~FP_Tree();
    FP_Tree(vector<string>,int);
    void Insert(vector<string>);
private:
    FP_Node * root;
};

void FP_Tree::Insert(vector<string> content)
{
    FP_Node * pre, temp;
    pre = new FP_Node;
    temp = new FP_Node;
    pre = root;
    vector<FP_Node *> * next_list;
    next_list = &(root->sub_list);
    if((*next_list).size())
        temp = next_list[0];
    else
    {
        FP_Node * t;
        t = new FP_Node;
        t->content = "empty";
        t->fre = 0;
        next_list->push_back(t);
        temp = t;
    }
    //一开始pre是root,next_list则是sub_list，temp则是sub_list[j]负责探路的
    //进入迭代后，先检查哪个sub_list[j]为当前content[i]的节点，若找到，则pre进入该节点并给该节点counter+1，next_list与temp跟上。
    //若没有，则在next_list内push_back一个新的分支然后再进入
    bool flag = false;
    for(int i = 0; i < content.size(); i++)
    {
        string curr_str = content[i];
        //先检查其是否为频繁单项
        
        
        for(int j = 0; j < pre->sub_list->size(); j++)
        {
            if(pre->sub_list[j]->content == curr_str)
            {
                flag = true;
                temp = pre->sub_list[j];
                break;
            }
        }
        if(flag)    //找到了当前匹配模式符合的节点。
        {
            temp->fre += 1;
            pre = temp;
            vector<FP_Node *> * next_inlist;
            next_inlist = &(pre->sub_list);
            if((*next_inlist).size())
            {
                temp = next_inlist[0];
            }
            else
            {
                FP_Node * k;
                k = new FP_Node;
                k->content = "empty";
                k->fre = 0;
                next_inlist->push_back(k);
                temp = k;
            }
        }
        else        //另开一个分支
        {
            FP_Node * new_branch;
            new_branch = new FP_Node;
            new_branch->content = curr_str;
            new_branch->fre = 1;
            pre->sub_list->push_back(new_branch);
            pre = new_branch;
            
            FP_Node * new_subbranch;
            new_subbranch = new FP_Node;
            new_subbranch->content = "empty";
            new_subbranch->fre = 0;
            pre->sub_list->push_back(new_subbranch);
            temp = new_subbranch;
        }
    }
}

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

bool Header_Linklist::Match(string v)
{
    Header * temp;
    temp = new Header;
    temp = head->next;
    while(temp)
    {
        if(temp->s == v)    return true;
        temp = temp->next;
    }
    return false;
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
        while(f[j] <= temp && j > i)    j--;
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

vector<string> Return_DESC_Content(string pre, vector<string> s, vector<int> f)
{
    vector<string> post;
    for(int i = 0; i < pre.size(); i++)
    {
        int max_index = 0;
        int max_fre = INT_MIN;
        for(int j = 0; j < pre.size(); j++)
        {
            int loc = Find_String(post,pre[j]);
            if(loc == -1)
            {
                if(max_fre < f[loc])
                {
                    max_index = loc;
                    max_fre = f[loc];
                }
            }
            else continue;
        }
        post.push_back(s[max_index]);
    }
    return post;
}

//s是存储频繁单项内容的数组，f是存储其出现次数的数组
void Find_MostFreq_SingleItem(string filename, vector<string> & s, vector<int> & f,Header_Linklist * hl, FP_Tree * fp_tree)
{
    //Search file and calculate each single feature's apperance,and sort them into two vectors
    ifstream fread(filename);
    string feature;
    vector<string> content,data;
    string d = ",";
    while(fin>>content)
    {
        data.push_back(content);
        //拆分content(,)然后得到feature(单个)       以后可以变成不定长度的feature
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
    
    InsertData_FormFP_Tree(s,f,fp_tree,data,hl);
    
}

void Delete_NonFreItem(vector<string> * p, Header_Linklist * hl)
{
    vector<string> result;
    for(int i = 0; i < p.size(); i++)
    {
        if(hl->Match(p[i]))     continue;
        else
        {
            p[i] = "-1"；
        }
    }
    for(int i = 0; i < p.size();i++)
    {
        if(p[i] != "-1")    result.push_back(p[i]);
        else continue;
    }
    p = &result;
}

void InsertData_FormFP_Tree(vector<string>s, vector<int> f, FP_Tree * fp_tree, vector<stirng> data, Header_Linklist * hl)
{
    //对照原数据将出现在其中的特征按总频率降序插入到root下生成初始FP树
    for(int i = 0; i < data.size(); i++)
    {
        //先将contant变成按feature降序排列的string
        vector<string> DESC_Content = Return_DESC_Content(data[i],s,f);
        //将DESC_Content中非频繁单项去除
        Delete_NonFreItem(&DESC_Content,hl);
        //再从root开始往下插入
        fp_tree->Insert(DESC_Content);
    }
}

//为每个频繁单项在初始FP树中从对应节点（多个）到root之间的路径的所有点做成一个各自特有的条件FP树（但是仅拔出在这棵树中最深的内容为该频繁单项的点）



//从条件FP树中不断抽出节点与之前以抽出的点结合成为新的频繁项



























