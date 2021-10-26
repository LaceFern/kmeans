#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

//fvecs2float
int fvecs_read (const char *fname, int d, int n, float *a){
  FILE *f = fopen (fname, "r");
  if (!f) {
    fprintf (stderr, "fvecs_read: could not open %s\n", fname);
    perror ("");
    return -1;
  }

  long i;
  for (i = 0; i < n; i++) {
    int new_d;

    if (fread (&new_d, sizeof (int), 1, f) != 1) {
      if (feof (f))
        break;
      else {
        perror ("fvecs_read error 1");
        fclose(f);
        return -1;
      }
    }

    if (new_d != d) {
      fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
      fclose(f);
      return -1;
    }

    if (fread (a + d * (long) i, sizeof (float), d, f) != d) {
      fprintf (stderr, "fvecs_read error 3\n");
      fclose(f);
      return -1;
    }
  }
  fclose (f);

  return i;
}


//svm2float
void libsvm_read(const char* fname, uint32_t n, uint32_t d, float* a) {
  std::string line;
  std::ifstream f(fname);

  int index = 0;
  if (f.is_open()){
    while(index < n){
      std::getline(f, line);
      int pos0 = 0;
      int pos1 = 0;
      int pos2 = 0;
      int column = 0;

      while(pos2 != -1){
        if (pos2 == 0) {
          pos2 = line.find(" ", pos1);
          float temp = std::stof(line.substr(pos1, pos2-pos1), NULL);
        }
        else{
          pos0 = pos2;
          pos1 = line.find(":", pos1) + 1;
          if(pos1 == 0){
            break;
          }
          // std::cout<<"pos:"<<pos1<<"\n";
          pos2 = line.find(" ", pos1);
          column = std::stof(line.substr(pos0 + 1, pos1 - pos0 - 1));
          if(column >= d){
            break;
          }
          if (pos2 == -1) {
            pos2 = line.length()+1;
            a[index * d + column-1] = std::stof(line.substr(pos1, pos2 - pos1), NULL);
          }
          else{
            a[index * d + column-1] = std::stof(line.substr(pos1, pos2 - pos1), NULL);
          }
          // std::cout << "a: "  << column << "\n";
          //std::cout << "index*d + column: "  << index*d + column-1 << "\n";
          //std::cout << "a[index*d + column]: "  << a[index*d + column-1] << "\n";
        }
      }
      index++;
    }
    f.close();
  }
  else
    std::cout << "Unable to open file " << fname << "\n";
  std::cout << "in libsvm, numSamples: "  << n << "\n";
  std::cout << "in libsvm, numFeatures: " << d << "\n"; 
}
