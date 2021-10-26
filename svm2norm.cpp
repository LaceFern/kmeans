#include <vector>
#include <stdint.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <fstream>

float* dr_a; //sample
float* dr_b; //label

void load_libsvm_data(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures, uint32_t _numBits);

int main(){
    char* pathToFile = "/home/zxy/dataset/avazu-app";
    uint32_t _numSamples = 40428967;
    uint32_t _numFeatures = 1000000;
    load_libsvm_data(pathToFile, _numSamples, _numFeatures, 32); 

    FILE *f=fopen("./avazu-app.txt","wt");
    if(f==NULL){
        printf("文件打开失败!\n");
    }
    else{
        printf("文件打开成功!\n");
        for(int i = 0; i < _numSamples; i++){
            for(int j = 0; j < _numFeatures; j++){
                fprintf(f,"%f ",dr_a[i * _numFeatures + j]);
            }
            fprintf(f,"\n");
        }
    }
}


void load_libsvm_data(char* pathToFile, uint32_t _numSamples, uint32_t _numFeatures, uint32_t _numBits) {
  std::cout << "Reading " << pathToFile << "\n";

  uint32_t _numFeatures_algin = ((_numFeatures+63)&(~63));

  dr_a  = (float*)malloc(_numSamples*_numFeatures*sizeof(float)); 
  std::cout<<_numSamples*_numFeatures<<"\n";
  if (dr_a == NULL)
  {
    printf("Malloc dr_a failed in load_tsv_data\n");
    return;
  }
  std::cout << "dra " << "\n";
  //////initialization of the array//////
  for (long i = 0; i < _numSamples*_numFeatures; i++){
    dr_a[i] = 0.0;
  }

  std::cout << "draa " << "\n";
  dr_b  = (float*)malloc(_numSamples*sizeof(float));
  if (dr_b == NULL)
  {
    printf("Malloc dr_b failed in load_tsv_data\n");
    return;
  }
  std::cout << "drb " << "\n";
  std::cout << "check-point0 " << "\n";
  std::string line;
  std::cout << "check-point1 " << "\n";
  std::ifstream f(pathToFile);
  std::cout << "check-point2 " << "\n";

  int index = 0;
  if (f.is_open()) 
  {
    while( index < _numSamples ) 
    {
      // std::cout<<index<<"\n";
      std::getline(f, line);
      int pos0 = 0;
      int pos1 = 0;
      int pos2 = 0;
      int column = 0;
      while ( pos2 != -1 ) //-1 (no bias...) //while ( column < _numFeatures ) 
      {
        if (pos2 == 0) 
        {
          
          pos2 = line.find(" ", pos1);
          float temp = std::stof(line.substr(pos1, pos2-pos1), NULL);
          
          dr_b[index] = temp;
          // std::cout << "dr_b: "  << temp << "\n";
        }
        else 
        {
          pos0 = pos2;
          pos1 = line.find(":", pos1)+1;
          if(pos1==0){
            break;
          }
          // std::cout<<"pos:"<<pos1<<"\n";
          pos2 = line.find(" ", pos1);
          column = std::stof(line.substr(pos0+1, pos1-pos0-1));
          if(column >= _numFeatures){
            break;
          }
          if (pos2 == -1) 
          {
            pos2 = line.length()+1;
            dr_a[index*_numFeatures + column-1] = std::stof(line.substr(pos1, pos2-pos1), NULL);
          }
          else{
            dr_a[index*_numFeatures + column-1] = std::stof(line.substr(pos1, pos2-pos1), NULL);
          }
          // std::cout << "dr_a: "  << column << "\n";
          //std::cout << "index*_numFeatures + column: "  << index*_numFeatures + column-1 << "\n";
          //std::cout << "dr_a[index*_numFeatures + column]: "  << dr_a[index*_numFeatures + column-1] << "\n";
        }
      }
      index++;
    }
    f.close();
  }
  else
    std::cout << "Unable to open file " << pathToFile << "\n";

  std::cout << "in libsvm, numSamples: "  << _numSamples << "\n";
  std::cout << "in libsvm, numFeatures: " << _numFeatures << "\n"; 
  std::cout << "in libsvm, _numFeatures_algin: " << _numFeatures_algin << "\n"; 
}

