#include "sc.h"
#include <algorithm>
#include <stack>
#include<cmath>

using namespace cv;
using namespace std;


bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image){

    if(new_width>in_image.cols){
        cout<<"Invalid request!!! new_width has to be smaller than the current size!"<<endl;
        return false;
    }
    if(new_height>in_image.rows){
        cout<<"Invalid request!!! new_height has to be smaller than the current size!"<<endl;
        return false;
    }
    
    if(new_width<=0){
        cout<<"Invalid request!!! new_width has to be positive!"<<endl;
        return false;

    }
    
    if(new_height<=0){
        cout<<"Invalid request!!! new_height has to be positive!"<<endl;
        return false;
        
    }
    
    return seam_carving_trivial(in_image, new_width, new_height, out_image);
}


// seam carves by removing trivial seams
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image){

    Mat iimage = in_image.clone();
    Mat oimage = in_image.clone();
    while(iimage.rows!=new_height || iimage.cols!=new_width){

        if(iimage.rows>new_height){
            reduce_horizontal_seam_trivial(iimage, oimage);
            iimage = oimage.clone();
        }
        
        if(iimage.cols>new_width){
            reduce_vertical_seam_trivial(iimage, oimage);
            iimage = oimage.clone();
        }
    }
    
    out_image = oimage.clone();
    return true;
}



bool reduce_horizontal_seam_trivial(Mat& in_image, Mat& out_image){


    Mat in_image_gray;
    Mat inn_image;
    Mat xgradient, ygradient;
    Mat weightmat;
    Mat xabs, yabs;

    GaussianBlur( in_image, inn_image, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor( inn_image, in_image_gray, CV_BGR2GRAY);
    Sobel( in_image_gray, xgradient, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    Sobel( in_image_gray, ygradient, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );

    //Laplacian( in_image_gray, xgradient, CV_16S, 3, 1, 0, BORDER_DEFAULT );
    //convertScaleAbs( xgradient, weightmat );


    convertScaleAbs( xgradient, xabs );
    convertScaleAbs( ygradient, yabs );

    addWeighted(xabs , 0.5, yabs, 0.5, 0, weightmat);

    int rows = in_image.rows;
    int cols = in_image.cols;
    
    int energyarray[rows][cols];

    out_image = Mat(rows-1,cols, CV_8UC3);
  
 
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            energyarray[i][j]=weightmat.at<uchar>(i,j);

        }
    }


//horizontal kay liye
    for(int i=1;i<cols;i++){
        for(int j=0;j<rows;j++){
      
            if(j==0){
                energyarray[j][i]= energyarray[j][i]+min(energyarray[j][i-1],energyarray[j+1][i-1]);

            }

            else if(j==rows-1){
                energyarray[j][i]=energyarray[j][i]+min(energyarray[j-1][i-1],energyarray[j][i-1]);
                
            }
            
            else{
                
                int a=min(energyarray[j-1][i-1],energyarray[j][i-1]);
                int b=min(a,energyarray[j+1][i-1]);
                energyarray[j][i]=energyarray[j][i]+b;
                
            }
            
        }
    }



    int sizemat=rows*cols;
    
    vector<int> energy1d;
    energy1d.resize(sizemat);

    //2d array to 1d array
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            int loc=(i*cols)+j;
            energy1d[loc]=energyarray[i][j];
        }    

    } 


    stack <int> visitstack;

    int lastrowstart=sizemat-1;
    int mini=energy1d[lastrowstart];
    visitstack.push(lastrowstart);
    for( int j=lastrowstart-cols;j>=cols-1;j=j-cols){
        if(mini>energy1d[j]){
            if(!visitstack.empty()){
                visitstack.pop();
            }
            mini=energy1d[j];
            visitstack.push(j);
        }
    }


    int last;

    while(visitstack.size()<cols){

        last=visitstack.top();


        if(last>=0 && last<=cols){


            if(energy1d[last-1]<energy1d[last+(cols-1)]){
               visitstack.push(last-1);
            }

            else{
               visitstack.push(last+(cols-1));
            }

        }

        else if(last>=sizemat-cols && last<=sizemat-1){


            if(energy1d[last-1]<energy1d[last-(cols+1)]){
               
               visitstack.push(last-1) ;
            }
            else{

               visitstack.push(last-(cols+1));
            }

        }
        else{



                int up=last-1;


                int left=last+(cols-1);

                int right=last-(cols+1);


                int c=0;
                int d=0;
                c=min(energy1d[up],energy1d[left]);
                d= min(c,energy1d[right]);


                if(d==energy1d[right]){

                    visitstack.push(right);
                }
                else if(d==energy1d[up]){

                    visitstack.push(up);
                }
                else{

                    visitstack.push(left);
                }

        }
    }



    int up;
    while (!visitstack.empty()) { 

        for(int i=0;i<cols;i++){

                up=visitstack.top(); 

                visitstack.pop();
                int y_index = up/cols;

            for(int j=0;j<y_index;j++){

                int x_index = i;

                Vec3b pixel = in_image.at<Vec3b>(j,x_index);  
                out_image.at<Vec3b>(j,x_index) =pixel;            
            }

            for(int j=y_index+1;j<rows;j++){

                int x_index = i;

                Vec3b pixel = in_image.at<Vec3b>(j, x_index); 
                out_image.at<Vec3b>(j-1,x_index) =pixel;

            }


        }

    }

    return true;
}

// vertical trivial seam is a seam through the center of the image
bool reduce_vertical_seam_trivial(Mat& in_image, Mat& out_image){


    Mat in_image_gray;
    Mat inn_image;
    Mat xgradient, ygradient;
    Mat weightmat;
    Mat xabs, yabs;


    GaussianBlur( in_image, inn_image, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor( inn_image, in_image_gray, CV_BGR2GRAY);
    Sobel( in_image_gray, xgradient, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    Sobel( in_image_gray, ygradient, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );

    convertScaleAbs( xgradient, xabs );
    convertScaleAbs( ygradient, yabs );

    addWeighted(xabs , 0.5, yabs, 0.5, 0, weightmat);

    //Laplacian( in_image_gray, xgradient, CV_16S, 3, 1, 0, BORDER_DEFAULT);
    //convertScaleAbs( xgradient, weightmat );

    int rows = in_image.rows;
    int cols = in_image.cols;
    
    int energyarray[rows][cols];
    out_image = Mat(rows, cols-1, CV_8UC3);
  
 
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            energyarray[i][j]=weightmat.at<uchar>(i,j);

        }
    }

    for(int i=1;i<rows;i++){
        for(int j=0;j<cols;j++){


            if(j==0){
                energyarray[i][j]=energyarray[i][j]+min(energyarray[i-1][j],energyarray[i-1][j+1]);
            }

            else if(j==cols-1){
                energyarray[i][j]=energyarray[i][j]+min(energyarray[i-1][j-1],energyarray[i-1][j]);
              
            }
            
            else{
                
                int a=min(energyarray[i-1][j-1],energyarray[i-1][j]);
                int b=min(a,energyarray[i-1][j+1]);
                energyarray[i][j]=energyarray[i][j]+b;
             

            }
            
        }
    }


    int sizemat=rows*cols;
    //int energy1d[sizemat];
    vector<int> energy1d;
    energy1d.resize(sizemat);


    //2d array to 1d array
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            int loc=(i*cols)+j;
            energy1d[loc]=energyarray[i][j];
        }    

    } 

    //MINIMUM FROM LAST ROW
    stack <int> visitstack;
    //vector <int> s;
    int lastrowstart=sizemat-cols;
    int mini=energy1d[lastrowstart];
    visitstack.push(lastrowstart);
    for( int j=lastrowstart+1;j<sizemat;j++){
        if(mini>energy1d[j]){
            if(!visitstack.empty()){
                visitstack.pop();
            }
            mini=energy1d[j];
            visitstack.push(j);
        }
    }

    int last;

    while(visitstack.size()<rows){

        last=visitstack.top();

        if(last%cols==0){

            if(energy1d[last-cols]<energy1d[last-(cols-1)]){
               visitstack.push(last-cols) ;
            }
            else{
               visitstack.push(last-(cols-1));
            }

        }
        else if((last+1)%cols==0){

            if(energy1d[last-cols]<energy1d[last-(cols+1)]){

               visitstack.push(last-cols) ;
            }
            else{

               visitstack.push(last-(cols+1));
            }

        }
        else{


                int up=last-cols;

                int left=last-(cols+1);

                int right=last-(cols-1);

                int c=0;
                int d=0;
                c=min(energy1d[up],energy1d[left]);
                d= min(c,energy1d[right]);

                if(d==energy1d[right]){
 
                    visitstack.push(right);
                }
                else if(d==energy1d[up]){
 
                    visitstack.push(up);
                }
                else{
 
                    visitstack.push(left);
                }

        }
    }


    int up;
    while (!visitstack.empty()) { 

        for(int i=0;i<rows;i++){

                up=visitstack.top(); 
                visitstack.pop();
                int y_index = up/cols;
 
            for(int j=i*cols;j<up;j++){

                int x_index = j % cols;

                Vec3b pixel = in_image.at<Vec3b>(y_index,x_index);  
                out_image.at<Vec3b>(y_index,x_index) =pixel;            
            }

            for(int j=up+1;j<(i+1)*cols;j++){
 
                int x_index = j % cols;

                Vec3b pixel = in_image.at<Vec3b>(y_index, x_index); 
                out_image.at<Vec3b>(y_index,x_index-1) =pixel;

            }

        }

    }
    
    return true;
}