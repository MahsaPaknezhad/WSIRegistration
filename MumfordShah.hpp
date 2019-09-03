/*
 * MumfordShah.hpp
 *
 *  Created on: Jan 20, 2016
 *      Author: lohsy
 *
 */

#ifndef MUMFORDSHAH_HPP_
#define MUMFORDSHAH_HPP_

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<highgui.h> 
#include<cv.h>      

/**
 * Does MumfordShah segmentation.
 *
 * NOT USED. USING SYSTEM CALL INSTEAD
 *
 * A simple copy paste will not work.
 * Arrays are too large to be stored on call stack.
 * Need to dynamically create and store on heap.
 * public:
 *   double* dat;
 * Then in constructor:
 *   dat = new double [SIZE * SIZE];
 * Then in destructor
 *   delete[] dat;
 */
class MumfordShah {
public:
	MumfordShah (std::string imageFilename, double mu, double T,
			double anneal, int SEED, int mcs, std::string check);

private:
	static const int SIZE = 300;
	static const int iqq = 3;
	static const int nblock = 4;

	std::string windowname;
	double dat[SIZE * SIZE];

	double original0[SIZE][SIZE][nblock + 1];
	double original[SIZE][SIZE];
	int spin0[SIZE][SIZE];
	int spin[SIZE][SIZE];
	IplImage * img;

	int outline[2 * SIZE][SIZE];
	int line_x[SIZE * SIZE], line_y[SIZE * SIZE];

	int S_Number[iqq], NewS_Number[iqq];
	int Cont_Number, NewC_Number;
	double O_Sum[iqq], NewO_Sum[iqq];

	int mcs, N;

	int w1, w2, Inte;

	double C[iqq], NewC[iqq];
	int sp_c[iqq], sp_ci[iqq];

	double mu, T;
	double anneal;
	int SEED;
	std::string check;

	double E, Ec, Eh;

	double mu_org;

	double p; // ratio of cluster growth

	void main_part(char *,int,int,int);
	void spin_initial(int,int);
	int scanning2(int,int);
	double HistE_initial(int,int);
	void metropolis(int,int);
	void rewrite(int,int,int,int,int,int);
	int Inline_UD(int,int,int,int,int,int);
	double dHistE(double,int,int);

	void print_original(char *,int,int,int);
	void print_segments(char *,int,double,int);
	void display_segments(int,int,int);
	void display_original(char *);

	void egginitial(int, int, double *);

	void loadBMP(char *,int *,int *,int *, double *);
	void saveBMP(char *,int ,int ,int ,double *);

	void sorting(void);
	//compare function must be static to be access by qsort
	static int doublecmp(const void *v1, const void *v2);

	void str_out(char *, char *, char *);
};

// -----------------------------------------------------
MumfordShah::MumfordShah(std::string imageFilename, double mu, double T,
		double anneal, int SEED, int mcs, std::string check) {

	//init
	windowname = "segments";
	Cont_Number=0;
	NewC_Number = 0;
	mcs = 1000;
	mu = 200.0;
	T = 200.0;
	anneal = 0.95;
	SEED = 12345;
	E = 0;
	Ec = 0;
	Eh = 0;
	p = 0.8;

	//init 2
	int cm, ht, wd;
	int i, j;
	int ht_org, wd_org;

	unsigned int flag;

	int block;
	char *str = "_out.bmp";
	char tmp[30];

	char *fname = new char[imageFilename.length() + 1];
	strcpy(fname, imageFilename.c_str());
	loadBMP(fname, &cm, &ht, &wd, dat);

	CvSize imgsize;
	imgsize.width = wd;
	imgsize.height = ht;
	img = cvCreateImage(imgsize, IPL_DEPTH_8U, 1);

	for (i = 0; i < ht; i++) {
		for (j = 0; j < wd; j++) {
			original0[j][i][0] = dat[i * wd + j];
		}
	}
	ht_org = ht;
	wd_org = wd;
	print_original(fname, ht, wd, 0);

	this->mu = mu;
	this->T = T;
	this->anneal = anneal;
	this->SEED = SEED;
	this->mcs = mcs;
	this->check = check;

	printf("# ht= %d wd= %d q= %d\n", ht, wd, iqq);
//	printf("### mu= %f T= %f anneal= %6.4f SEED= %d MCS= %d ratio of cluster= %4.2f check=%s\n",
//	            mu,T,anneal,SEED,mcs,p,check);

	std::cout<<"### mu= "<<mu<<" T= "<<T<<" anneal= "<<anneal<<" SEED= "<<SEED<<" MCS= "<<mcs<<" ratio of cluster= "
			<<p<<" check= "<<check<<std::endl;

	  if(check.compare("yes")==0){
	    flag=0;
	  }else{
	    flag=1;
	  }

	printf("# imcs, T, E, Etrue, Ec, Eh, Cn\n");

	if(flag==0){
	    cvNamedWindow(windowname.c_str(),CV_WINDOW_AUTOSIZE);
	    display_original(fname);
	  }

	for (block = 1; block <= nblock; block++) {
		for (i = 0; i < ht / 2; i++) {
			for (j = 0; j < wd / 2; j++) {
				original0[j][i][block] = (dat[2 * i * wd + 2 * j]
						+ dat[(2 * i + 1) * wd + 2 * j]
						+ dat[2 * i * wd + 2 * j + 1]
						+ dat[(2 * i + 1) * wd + 2 * j + 1]) / 4;
			}
		}

		ht = ht / 2;
		wd = wd / 2;

		print_original(fname, ht, wd, block);

		for (i = 0; i < ht; i++) {
			for (j = 0; j < wd; j++) {
				dat[i * wd + j] = original0[j][i][block];
			}
		}
	}

	for (i = 0; i < ht; i++) {
		for (j = 0; j < wd; j++) {
			original[j][i] = original0[j][i][nblock];
		}
	}

	mu_org = mu;
	mu = mu / pow(2, nblock);
	T = T / pow(2, nblock);

	spin_initial(ht, wd);

	main_part(fname, ht, wd, flag);

	for (i = 0; i < ht_org; i++) {
		for (j = 0; j < wd_org; j++)
			dat[i * wd_org + j] = sp_c[spin[j][i]] * Inte;
	}

	str_out(fname, str, tmp);
	print_segments(fname, 0, T, mcs);
	saveBMP(tmp, 1, ht_org, wd_org, dat);
}

void MumfordShah::main_part(char * basename, int ht,int wd,int flag){

  int i,j,imcs;
  int block;
  int block_change=0;

  block=nblock;
  printf("# block= %d ht= %d wd= %d mu= %f\n",block,ht,wd,mu);

  for(imcs=1;imcs<=mcs;imcs++){

   // if((rand()/(RAND_MAX+1.0))<p){
   //   metropolis2(ht,wd);   // one sweep through lattice - some bug in boundary updating
   //   printf("metropolis2 called");
   // }else{
      metropolis(ht,wd);
      printf("metropolis called");
   // }
    //printf("%s %d E now is %f\n",__FILE__,__LINE__,E);

    if(imcs%10==0){
      if(flag==0){
	display_segments(ht,wd,block);
      } else {
	// check energy

        int chk_Cont_Number=scanning2(ht,wd);
        double chk_Ec=mu*chk_Cont_Number;
        double chk_Eh=HistE_initial(ht, wd);
	double chk_E = chk_Ec+chk_Eh;

	if(fabs(chk_E-E)>1e-3) {
	  printf("%s %d error in energy %f %f diff %f ",__FILE__,__LINE__,chk_E,E,chk_E-E);
	  exit(1);
	}

        printf("%s%d imcs %d T %f E/1000 %f %f %f %f %d\n",__FILE__,__LINE__,
        imcs,T,E/1000,(mu_org/mu*Ec+Eh)/1000,Ec/1000,Eh/1000,Cont_Number);
      }
      T=T*anneal;
    }
 //   mu=mu/anneal;
 //   T=T/pow(anneal,.7);

    if(imcs%20==0){
   // if(imcs%100==0){
      if(flag==0){
        printf("imcs= %d T= %f E= %f Etrue= %f Ec= %f Eh= %f Cn= %d mu=%f\n",
        imcs,T,E/1000,(mu_org/mu*Ec+Eh)/1000,Ec/1000,Eh/1000,Cont_Number,mu);
      }
    }
//
//    change of block level
//
    double factor=1.75;
    if(block==8 && (imcs>=mcs/pow(factor,8))) {
      block=7;
      block_change = 1;
    }
    else if(block==7 && (imcs>=mcs/pow(factor,7))) {
      block=6;
      block_change = 1;
    }
    else if(block==6 && (imcs>=mcs/pow(factor,6))) {
      block=5;
      block_change = 1;
    }
    else if(block==5 && (imcs>=mcs/pow(factor,5))) {
      block=4;
      block_change = 1;
    }
    else if(block==4 && (imcs>=mcs/pow(factor,4))) {
      block=3;
      block_change = 1;
    }
    else if(block==3 && (imcs>=mcs/pow(factor,3))) {
      block=2;
      block_change = 1;
    }
    else if(block==2 && (imcs>=mcs/pow(factor,2))) {
      block=1;
      block_change = 1;
    }else if(block==1 && (imcs>=mcs/factor)) {
      block=0;
      block_change = 1;
    }else{
      block_change = 0;
    }

    if(block_change==1) {

      print_segments(basename,block+1,T,imcs);

      mu=mu*2;
      T=T*2;

      for(i=0;i<ht;i++){
        for(j=0;j<wd;j++) {
          spin0[j][i]= spin[j][i];
        }
      }

      for(i=0;i<ht;i++){
        for(j=0;j<wd;j++) {
          spin[2*j][2*i]= spin0[j][i];
          spin[2*j+1][2*i]= spin0[j][i];
          spin[2*j][2*i+1]= spin0[j][i];
          spin[2*j+1][2*i+1]= spin0[j][i];
        }
      }

      ht=ht*2; wd=wd*2;
      N=ht*wd;

      for(i=0;i<ht;i++){
        for(j=0;j<wd;j++) {
          original[j][i]= original0[j][i][block];
        }
      }

      Cont_Number=scanning2(ht,wd);
      Ec=mu*Cont_Number;
      Eh=HistE_initial(ht, wd);
      E=Ec+Eh;

      printf("# block= %d ht= %d wd= %d mu= %f\n",block,ht,wd,mu);
    }
    block_change = 0; // reset
  }

  sorting();

  printf("# ");
  for(i=0;i<iqq;i++){printf("s_n[%d]= %d ",i,S_Number[sp_ci[i]]);}
  printf("\n");

  printf("# ");
  for(i=0;i<iqq;i++){printf("c_n[%d]= %f ",i,C[sp_ci[i]]);}
  printf("\n");

}

// -----------------------------------------------------
void MumfordShah::spin_initial(int ht,int wd){

//  set initial spins and energy

  int i,j;

  Inte=(int)(255/(iqq-1));
  srand(SEED);

  for(i=0;i<wd;i++){
    for(j=0;j<ht;j++){
       spin[i][j]=rand()%iqq;
    }
  }

  N=ht*wd;
  Cont_Number=scanning2(ht,wd);
  Ec=mu*Cont_Number;
  Eh=HistE_initial(ht, wd);
  E=Ec+Eh;
}

// -----------------------------------------------------
int MumfordShah::scanning2(int ht, int wd){

//  compute Cont_Number (length of contour) (cluster growth)

  int i,j;
  int cont=0;

  for(i=0;i<wd-1;i++){
    for(j=0;j<ht;j++){
      if(spin[i][j]!=spin[i+1][j]){
        cont++;
   //
   //            horizontal direction
   //
        outline[i][j]=cont;
        line_x[cont]=i; line_y[cont]=j;
   //
      }
    }
  }

  for(i=0;i<wd;i++){
    for(j=0;j<ht-1;j++){
      if(spin[i][j]!=spin[i][j+1]){
        cont++;
   //
   //            vertical direction
   //
        outline[i+wd][j]=cont;
        line_x[cont]=i+wd; line_y[cont]=j;
   //
      }
    }
  }
  return(cont);
}

// -----------------------------------------------------
double MumfordShah::HistE_initial(int ht, int wd){

//  compute initial E_hist

  int i,j;
  double HistE[iqq],sum=0;

  for(i=0;i<iqq;i++){
      S_Number[i]=0;
      O_Sum[i]=0;
      HistE[i]=0;
  }

  for(i=0;i<wd;i++){
    for(j=0;j<ht;j++){
        S_Number[spin[i][j]]=S_Number[spin[i][j]]+1;
        O_Sum[spin[i][j]]=O_Sum[spin[i][j]]+original[i][j];
    }
  }

  for(i=0;i<iqq;i++) C[i]=O_Sum[i]/(double)S_Number[i];

  for(i=0;i<wd;i++){
    for(j=0;j<ht;j++){
    HistE[spin[i][j]]=HistE[spin[i][j]]
         +(original[i][j]-C[spin[i][j]])*(original[i][j]-C[spin[i][j]]);
    }
  }

  for(i=0;i<iqq;i++) sum=sum+HistE[i];

  return(sum);

}

// -----------------------------------------------------
void MumfordShah::metropolis(int ht,int wd){

//  operation for single MCS

  int i,x,y,spnew,spold;
  double dE=0,dEc=0,dEh=0;

  for(i=0;i<N;i++){

    x=(int)(wd*(rand()/(RAND_MAX+1.0)));
    y=(int)(ht*(rand()/(RAND_MAX+1.0)));

    spold=spin[x][y];
    spnew=(spin[x][y]+rand()%(iqq-1)+1)%iqq;

    if(S_Number[spold]<=1) {
      // do nothing
    }
    else {
      NewC_Number=Cont_Number+Inline_UD(x,y,spnew,spold,wd,ht);
      dEc=mu*(NewC_Number-Cont_Number);
      dEh=dHistE(original[x][y],spnew,spold);
      dE=dEh+dEc;

      if((rand()/(RAND_MAX+1.0))<=exp(-dE/T)){ // accept
         Ec=Ec+dEc;
         Eh=Eh+dEh;
         E=Eh+Ec;
         Cont_Number=NewC_Number;
         spin[x][y]=spnew;
         S_Number[spold]=NewS_Number[spold]; S_Number[spnew]=NewS_Number[spnew];
         C[spold]=NewC[spold]; C[spnew]=NewC[spnew];
         O_Sum[spold]=NewO_Sum[spold]; O_Sum[spnew]=NewO_Sum[spnew];
      }
    }
  }
}


// -----------------------------------------------------
void MumfordShah::rewrite(int x, int y, int spnew, int spold, int ht, int wd){

//  update of bond information

    int line_max;
    int line_xmax, line_ymax, outl;

    line_max=Cont_Number;

    if(x!=wd-1){
      if(spin[x+1][y]==spold){
        line_max=line_max+1;
        outline[x][y]=line_max;
        line_x[line_max]=x; line_y[line_max]=y;
      }else{
        if(spin[x+1][y]==spnew){
          outl=outline[x][y];
          line_xmax=line_x[line_max]; line_ymax=line_y[line_max];
          outline[line_xmax][line_ymax]=outl;
          line_x[outl]=line_xmax;
          line_y[outl]=line_ymax;
          outline[x][y]=0; line_max=line_max-1;
        }
      }
    }

    if(x!=0){
      if(spin[x-1][y]==spold){
        line_max=line_max+1;
        outline[x-1][y]=line_max;
        line_x[line_max]=x-1; line_y[line_max]=y;
      }else{
        if(spin[x-1][y]==spnew){
          outl=outline[x-1][y];
          line_xmax=line_x[line_max]; line_ymax=line_y[line_max];
          outline[line_xmax][line_ymax]=outl;
          line_x[outl]=line_xmax;
          line_y[outl]=line_ymax;
          outline[x-1][y]=0; line_max=line_max-1;
        }
      }
    }

    if(y!=ht-1){
      if(spin[x][y+1]==spold){
        line_max=line_max+1;
        outline[x+wd][y]=line_max;
        line_x[line_max]=x+wd; line_y[line_max]=y;
      }else{
        if(spin[x][y+1]==spnew){
          outl=outline[x+wd][y];
          line_xmax=line_x[line_max]; line_ymax=line_y[line_max];
          outline[line_xmax][line_ymax]=outl;
          line_x[outl]=line_xmax;
          line_y[outl]=line_ymax;
          outline[x+wd][y]=0; line_max=line_max-1;
        }
      }
    }

    if(y!=0){
      if(spin[x][y-1]==spold){
        line_max=line_max+1;
        outline[x+wd][y-1]=line_max;
        line_x[line_max]=x+wd; line_y[line_max]=y-1;
      }else{
        if(spin[x][y-1]==spnew){
          outl=outline[x+wd][y-1];
          line_xmax=line_x[line_max]; line_ymax=line_y[line_max];
          outline[line_xmax][line_ymax]=outl;
          line_x[outl]=line_xmax;
          line_y[outl]=line_ymax;
          outline[x+wd][y-1]=0; line_max=line_max-1;
        }
      }
    }

  }

// -----------------------------------------------------
int MumfordShah::Inline_UD(int x,int y,int spnew,int spold,int wd, int ht){

//  change of Cont_Number

   int cont=0;

   if(x!=wd-1){
     if(spin[x+1][y]==spnew){cont--;}
       else {if(spin[x+1][y]==spold) {cont++;} }
   }
   if(x!=0){
     if(spin[x-1][y]==spnew){cont--;}
       else {if(spin[x-1][y]==spold) {cont++;} }
   }
   if(y!=ht-1){
     if(spin[x][y+1]==spnew){cont--;}
       else {if(spin[x][y+1]==spold) {cont++;} }
   }
   if(y!=0){
     if(spin[x][y-1]==spnew){cont--;}
       else {if(spin[x][y-1]==spold) {cont++;} }
   }

   return(cont);
}

// -----------------------------------------------------
double MumfordShah::dHistE(double orig,int spnew,int spold){

//  change of E_hist

  NewO_Sum[spold]=O_Sum[spold]-orig;
  NewO_Sum[spnew]=O_Sum[spnew]+orig;

  NewS_Number[spold]=S_Number[spold]-1;
  NewS_Number[spnew]=S_Number[spnew]+1;

  if(NewS_Number[spnew]<=0) {
    printf("NewS_Number[spnew] = %d\n",NewS_Number[spnew]);
    throw;
  }
  if(NewS_Number[spold]<=0) {
    printf("NewS_Number[spold] = %d\n",NewS_Number[spold]);
    throw;
  }
  if(NewO_Sum[spold]<0) {
    printf("NewO_Sum[spold] = %f\n",NewO_Sum[spold]);
    throw;
  }
  if(NewO_Sum[spnew]<0) {
    printf("NewO_Sum[spnew] = %f\n",NewO_Sum[spnew]);
    throw;
  }
  NewC[spold]=NewO_Sum[spold]/(double)(NewS_Number[spold]);
  NewC[spnew]=NewO_Sum[spnew]/(double)(NewS_Number[spnew]);

  return(-NewC[spold]*NewO_Sum[spold]+C[spold]*O_Sum[spold]
         -NewC[spnew]*NewO_Sum[spnew]+C[spnew]*O_Sum[spnew]);
}

// -----------------------------------------------------
void MumfordShah::loadBMP(char *loadfile,int *s,int *ht,int *wd,double *dat){

  FILE *fr;
  char fname[80];
  unsigned char h[54];
  int width,height,cm;
  int pal[1024],pad,dm;
  int j,k,c,m;
  unsigned char dummy=0;

  strcpy(fname,loadfile);
  if(strstr(fname,".bmp")==NULL) strcat(fname, ".bmp");
  fr=fopen(fname,"rb");
  if(fr==NULL){
    printf("file is missing\n"); exit(-1);
  }
  for(j=0; j<54; j++) h[j]=fgetc(fr);
  if(!(h[0]=='B' && h[1]=='M')){
    printf("not BMP file\n"); exit(-1);
  }
  width=h[18]+h[19]*256; height=h[22]+h[23]*256;
  cm=h[28]/8;
  if(cm==1)for(j=0;j<1024;j++) pal[j]=fgetc(fr);
  pad=((4-(width%4))*cm)%4;

  for(j=0; j<height; j++){
    for(k=0; k<width; k++){
      if(cm==1){
        *(dat++)=pal[fgetc(fr)*4];
      } else {
        for(c=0; c<cm; c++) dummy=fgetc(fr);
          *(dat++)=dummy;
        }
      }
    for(m=0; m<pad; m++) dm=fgetc(fr);
  }

  fclose(fr);
  *s=cm; *ht=height; *wd=width;
}

// -----------------------------------------------------
void MumfordShah::saveBMP(char *filnam,int cm,int ht,int wd,double *datt){

  unsigned char h[54];
  int pad;
  int hdsz=54, fsiz;
  FILE *fw;
  int k,x,y,c;

  for(k=0; k<54; k++) h[k]=0;
  h[0]='B'; h[1]='M';
  pad=((4-(wd%4))*cm)%4;
  if(cm==1) hdsz+=1024;
  fsiz=(wd+pad)*ht*cm+hdsz;
  h[2] = fsiz%256; h[3] = fsiz/256; h[4] = fsiz/(256*256);
  h[10]= hdsz%256; h[11]= hdsz/256; h[12]= hdsz/(256*256);
  h[14]= 40;
  h[18]= wd % 256; h[19]= wd/256; h[20]=wd/(256*256);
  h[22]= ht % 256; h[23]= ht/256; h[24]=ht/(256*256);
  h[26]= 1; h[28]=cm*8;

  fw=fopen(filnam,"wb");
  for(k=0; k<54; k++) fputc(h[k], fw);
  if(cm==1) for(k=0; k<256; k++){
    fputc(k,fw); fputc(k,fw); fputc(k,fw); fputc(0,fw);
  }

  for(y=0; y<ht; y++){
    for(x=0; x<wd; x++){
      for(c=0; c<cm; c++) fputc((int)(datt[(y*wd+x)*cm+c]*.9),fw);
    }
    for(k=0; k<pad; k++) fputc(0,fw);
  }
  fclose(fw);
}

// -----------------------------------------------------
void MumfordShah::sorting(void){

  int sp,sp1;
  double C_sort[iqq];

  for(sp=0;sp<iqq;sp++){
    C_sort[sp]=C[sp];
  }

  qsort(C_sort,iqq,sizeof(C_sort[0]),doublecmp);
  for(sp=0;sp<iqq;sp++){
    for(sp1=0;sp1<iqq;sp1++){
      if(C_sort[sp]==C[sp1]){
        sp_c[sp1]=sp;
        sp_ci[sp]=sp1;
      }
    }
  }
}

// -----------------------------------------------------
int MumfordShah::doublecmp(const void *v1, const void *v2) {
   return (*(double *)v1 - *(double *)v2);
}

// -----------------------------------------------------
void MumfordShah::str_out(char *s1, char *s2, char *t)
{
  char tmp[30];
  int len=0,i=0,nn;

  while(*s1!='.'){
  tmp[len]=*s1++;
  len++;
  }
  nn=strlen(s2);

  for(i=0;i<nn;i++){
  tmp[len+i] = *s2++;
  }

  tmp[len+i]='\0';
  nn=strlen(tmp);
  i=0;
  while(i<nn){
  *t++=tmp[i];
  i++;
  }
  *t='\0';
}

// -----------------------------------------------------
void MumfordShah::print_segments(char * basename,int block,double T,int imcs) {
  char outname[500];
  sprintf(outname,"%s_block%d_T%f_mcs%d.bmp",basename,block,T,imcs);
  //HK2014 cvSaveImage(outname,img);
}
// -----------------------------------------------------
void MumfordShah::print_original(char * basename,int ht,int wd, int block) {

  int i,j,jp,dx,dy;
  double v;
  int two_pow_block=1;

  char outname[500];
  sprintf(outname,"%s_original_block%d.bmp",basename,block);

  for(i=0;i<block;++i) { two_pow_block *= 2; }
  CvSize s; s.width = two_pow_block*wd; s.height = two_pow_block*ht;

  IplImage * oimg = cvCreateImage(s,IPL_DEPTH_8U,1);
  for(i=0;i<wd;++i) {
    for(jp=0;jp<ht;++jp) {
      j = jp;
      v = original0[i][j][block];
      for(dy=0;dy<two_pow_block;++dy) {
        for(dx=0;dx<two_pow_block;++dx) {
	  cvSetReal2D(oimg,two_pow_block*j+dy,two_pow_block*i+dx,v);
	}
      }
    }
  }
  //HK2014 cvSaveImage(outname,oimg);
  cvReleaseImage(&oimg);
}
// -----------------------------------------------------
void MumfordShah::display_original(char * fname) {

  cvNamedWindow(fname,CV_WINDOW_AUTOSIZE);
  IplImage * oimg = cvLoadImage(fname,CV_LOAD_IMAGE_GRAYSCALE);
  cvShowImage("originalimage",oimg);
  cvWaitKey(0);
}
// -----------------------------------------------------
void MumfordShah::display_segments(int ht,int wd,int block) {

  int i,j,jp,dx,dy;
  double v;
  int two_pow_block=1;

  for(i=0;i<block;++i) { two_pow_block *= 2; }

  sorting();

  for(i=0;i<wd;++i) {
    for(jp=0;jp<ht;++jp) {

      j = jp;
      //switch(sp_c[spin[i][jp]]){
      //  case    0: v = 0             ; break;
      //  case    1: v =   225./(iqq-1); break;
      //  case    2: v = 2*225./(iqq-1); break;
      //  case    3: v = 3*225./(iqq-1); break;
      //  default: printf("error\n"); exit(1)  ;
      //}
      v = sp_c[spin[i][jp]]*225./(iqq-1);

      for(dy=0;dy<two_pow_block;++dy) {
        for(dx=0;dx<two_pow_block;++dx) {
	  cvSetReal2D(img,two_pow_block*j+dy,two_pow_block*i+dx,v);
	}
      }
    }
  }
  cvShowImage(windowname.c_str(),img);
  cvWaitKey(200);
}

#endif /* MUMFORDSHAH_HPP_ */
