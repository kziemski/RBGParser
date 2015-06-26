#include <cstdio>
#include <set>
#include <string>
#include <cmath>
#include <sstream>

#define maxl 10000
using namespace std;

set<string> word;

void read(string file)
{   
    char ch[maxl], form[maxl];
    int id;
    
    FILE *fp = fopen(file.c_str(),"r");
    
    while(!feof(fp)) {
        fgets(ch,maxl,fp);
        if (sscanf(ch,"%d %s",&id, form) != -1)
            word.insert(form);
    }
    
    fclose(fp);
}

int main()
{
    read("../PTB_SD_3_3_0/train.conll");
    read("../PTB_SD_3_3_0/test.conll");
    read("../PTB_SD_3_3_0/dev.conll");
    
//    read("data\\english.sample.train.lab");
    
    FILE *fp = fopen("../glove.840B.300d.txt", "r");
    FILE *fo = fopen("../glove.840B.300d.prune.txt", "w");
//    FILE *fp = fopen("data\\word_embedding","r");
//    FILE *fo = fopen("data\\word_embedding_prune","w");
    
    char ch[maxl], form[maxl];
    double v[maxl];
    
    while(!feof(fp)) {
        fgets(ch,maxl,fp);
        stringstream sio(ch);
        if (!(sio>>form) || word.find(form) == word.end())
            continue;
        
        int l = 0;
        double norm = 0;
        while (sio>>v[l]) {
            norm += v[l]*v[l];
            l++;
        }
        fprintf(fo,"%s",form);
        for (int i = 0; i < l; i++)
            fprintf(fo," %lf",v[i]/sqrt(norm));
        fprintf(fo,"\n");
    }
    
    fclose(fp);
    fclose(fo);
    
    return 0;
}
