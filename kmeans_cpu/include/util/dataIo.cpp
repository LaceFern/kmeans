
#include "dataIo.h"

void random_init(float *array, const size_t N, const size_t D){
  /*  short unsigned seed[3];
    int i,j;
    int max = 10;

    // seed[0]=1; seed[1]=1; seed[2]=2;

    //#pragma omp parallel for firstprivate(seed)
    for ( i=0 ; i < N  ; i++ ){
	for(j=0; j<D; j++){
            array[i*D+j] = (double)(rand()%max);
	    // printf("%f ", array[i*D+j]);
	}
	// printf("\n");
    }*/
   short unsigned seed[3];
    int i;

    seed[0]=1; seed[1]=1; seed[2]=2;

    #pragma omp parallel for firstprivate(seed)
    for ( i=0 ; i < N * D ; i++ ){
        array[i] = erand48(seed);
    }

}

void read_file(float *array, const size_t N, const size_t D, const char *filename, bool isBinary){
    FILE *fp;
    size_t counts = 0;
    size_t i=0,j=0;
    char line[MAX_LINE_LENGTH];
    char *token=NULL;
    const char space[2] = " ";

    fp = fopen(filename,"r");

    if ( fp == NULL ){
        fprintf(stderr, "File '%s' does not exists!", filename);
        exit(1);
    }

    if ( isBinary ){
        // read binary file, everything at once
        counts = fread(array, sizeof(float) * N * D, 1, fp);

        if ( counts == 0 ) {
            fprintf(stderr, "Binary file '%s' could not be read. Wrong format.", filename);
            exit(1);
        }
    }else{
        // processing a text file
        // format: there are D float values each line. Each value is separated by a space character.
        // notice MAX_LINE_LENGTH = 2049
        i = 0;
        while ( fgets ( line, MAX_LINE_LENGTH, fp ) != NULL &&
                i < N ) {


            if ( line[0] != '%'){ // ignore '%' comment char
                token = strtok(line, space);
                j=0;


                while ( token != NULL &&
                        j < D ){
                            
                    array[i*D + j] = atof(token); // 0.0 if no valid conversion
                    token = strtok(NULL, space);
                    j++;
                }
                i++;
            }
        }
    }

    fclose(fp);
}


void read_file_int(int *array, const size_t N, const size_t D, const char *filename, bool isBinary){
    FILE *fp;
    size_t counts = 0;
    size_t i=0,j=0;
    char line[MAX_LINE_LENGTH];
    char *token=NULL;
    const char space[2] = " ";

    fp = fopen(filename,"r");

    if ( fp == NULL ){
        fprintf(stderr, "File '%s' does not exists!", filename);
        exit(1);
    }

    if ( isBinary ){
        // read binary file, everything at once
        counts = fread(array, sizeof(float) * N * D, 1, fp);

        if ( counts == 0 ) {
            fprintf(stderr, "Binary file '%s' could not be read. Wrong format.", filename);
            exit(1);
        }
    }else{
        // processing a text file
        // format: there are D float values each line. Each value is separated by a space character.
        // notice MAX_LINE_LENGTH = 2049
        i = 0;
        while ( fgets ( line, MAX_LINE_LENGTH, fp ) != NULL &&
                i < N ) {


            if ( line[0] != '%'){ // ignore '%' comment char
                token = strtok(line, space);
                j=0;


                while ( token != NULL &&
                        j < D ){
                            
                    array[i*D + j] = atof(token); // 0.0 if no valid conversion
                    token = strtok(NULL, space);
                    j++;
                }
                i++;
            }
        }
    }

    fclose(fp);
}


void read_file_uint8(u_int8_t *array, const size_t N, const size_t D, const char *filename, bool isBinary){
    FILE *fp;
    size_t counts = 0;
    size_t i=0,j=0;
    char line[MAX_LINE_LENGTH];
    char *token=NULL;
    const char space[2] = " ";

    fp = fopen(filename,"r");

    if ( fp == NULL ){
        fprintf(stderr, "File '%s' does not exists!", filename);
        exit(1);
    }

    if ( isBinary ){
        // read binary file, everything at once
        counts = fread(array, sizeof(u_int8_t) * N * D, 1, fp);

        if ( counts == 0 ) {
            fprintf(stderr, "Binary file '%s' could not be read. Wrong format.", filename);
            exit(1);
        }
    }else{
        // processing a text file
        // format: there are D float values each line. Each value is separated by a space character.
        // notice MAX_LINE_LENGTH = 2049
        i = 0;
        while ( fgets ( line, MAX_LINE_LENGTH, fp ) != NULL &&
                i < N ) {


            if ( line[0] != '%'){ // ignore '%' comment char
                token = strtok(line, space);
                j=0;


                while ( token != NULL &&
                        j < D ){
                            
                    array[i*D + j] = atof(token); // 0.0 if no valid conversion
                    token = strtok(NULL, space);
                    j++;
                }
                i++;
            }
        }
    }

    fclose(fp);
}

void read_file_double(double *array, const size_t N, const size_t D, const char *filename, bool isBinary){
    FILE *fp;
    size_t counts = 0;
    size_t i=0,j=0;
    char line[MAX_LINE_LENGTH];
    char *token=NULL;
    const char space[2] = " ";

    fp = fopen(filename,"r");

    if ( fp == NULL ){
        fprintf(stderr, "File '%s' does not exists!", filename);
        exit(1);
    }

    if ( isBinary ){
        // read binary file, everything at once
        counts = fread(array, sizeof(double) * N * D, 1, fp);

        if ( counts == 0 ) {
            fprintf(stderr, "Binary file '%s' could not be read. Wrong format.", filename);
            exit(1);
        }
    }else{
        // processing a text file
        // format: there are D double values each line. Each value is separated by a space character.
        // notice MAX_LINE_LENGTH = 2049
        i = 0;
        while ( fgets ( line, MAX_LINE_LENGTH, fp ) != NULL &&
                i < N ) {


            if ( line[0] != '%'){ // ignore '%' comment char
                token = strtok(line, space);
                j=0;


                while ( token != NULL &&
                        j < D ){
                            
                    array[i*D + j] = atof(token); // 0.0 if no valid conversion
                    token = strtok(NULL, space);
                    j++;
                }
                i++;
            }
        }
    }

    fclose(fp);
}

void save_binary_file(float *array, const size_t N, const size_t D, char filename[]){
    FILE *fp=NULL;
    size_t counts = 0;

    fp = fopen(filename, "w");

    if ( fp == NULL ){
        fprintf(stderr, "Could not open file '%s'!", filename);
        exit(1);
    }

    counts = fwrite(array,sizeof(float) * N * D, 1, fp);

    if ( counts == 0 ){
        fprintf(stderr, "Error in writing file '%s'. Abort.", filename);
        exit(1);
    }

    fclose(fp);
}

void save_text_file(float *array, const size_t N, const size_t D, char filename[]){
    FILE *fp=NULL;
    size_t counts = 0;
    size_t i=0, j=0;
    char line[MAX_LINE_LENGTH];
    char strfloat[50];

    fp = fopen(filename, "w");

    if ( fp == NULL ){
        fprintf(stderr, "Could not open file '%s'!", filename);
        exit(1);
    }

    for ( i=0 ; i < N ; i++ ){
        strcpy(line, "");
        for ( j=0 ; j < D ; j++ ){
            strcpy(strfloat, "");
            sprintf(strfloat, "%f ", array[i*D + j]);
            strcat(line, strfloat);

        }
        fprintf(fp, "%s\n", line);

    }

    fclose(fp);
}


//fvecs2float
// int fvecs_read (const char *fname, int n, int d, float *a){
//   FILE *f = fopen (fname, "r");
//   if (!f) {
//     fprintf (stderr, "fvecs_read: could not open %s\n", fname);
//     perror ("");
//     return -1;
//   }

//   for (long long int i = 0; i < n; i++) {
//     int new_d;

//     if (fread (&new_d, sizeof (int), 1, f) != 1) {
//       if (feof (f))
//         break;
//       else {
//         perror ("fvecs_read error 1");
//         fclose(f);
//         return -1;
//       }
//     }

//     if (new_d != d) {
//       fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
//       fclose(f);
//       return -1;
//     }

//     if (fread (a + d * (long) i, sizeof (float), d, f) != d) {
//       fprintf (stderr, "fvecs_read error 3\n");
//       fclose(f);
//       return -1;
//     }
//   }
//   fclose (f);

//   return 1;
// }

int fvecs_read (const char *fname, int n, int d, float *a){
  FILE *f = fopen (fname, "r");
  if (!f) {
    fprintf (stderr, "fvecs_read: could not open %s\n", fname);
    perror ("");
    return -1;
  }

  for (long long int i = 0; i < n; i++) {
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

    // if (new_d != d) {
    //   fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
    //   fclose(f);
    //   return -1;
    // }

    if (fread (a + d * (long) i, sizeof (float), new_d, f) != new_d) {
      fprintf (stderr, "fvecs_read error 3\n");
      fclose(f);
      return -1;
    }
  }
  fclose (f);

  return 1;
}

//svm2float
int libsvm_read(const char* fname, int n, int d, float* a) {
  std::string line;
  std::ifstream f(fname);

  long long int index = 0;
  if (f.is_open()){
    while(index < n){
      std::getline(f, line);
      long long int pos0 = 0;
      long long int pos1 = 0;
      long long int pos2 = 0;
      long long int column = 0;

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
            a[1l * index * d + column-1] = std::stof(line.substr(pos1, pos2 - pos1), NULL); //这句话报错了
          }
          else{
            a[1l * index * d + column-1] = std::stof(line.substr(pos1, pos2 - pos1), NULL);
          }
          // std::cout << "a: "  << column << "\n";
          //std::cout << "index*d + column: "  << index*d + column-1 << "\n";
          //std::cout << "a[index*d + column]: "  << a[index*d + column-1] << "\n";
        }
      }
      index++;
    }
    f.close();
    return 0;
  }
  else{
      std::cout << "Unable to open file " << fname << "\n";
      return -1;
  }
    
  std::cout << "in libsvm, numSamples: "  << n << "\n";
  std::cout << "in libsvm, numFeatures: " << d << "\n"; 
}
