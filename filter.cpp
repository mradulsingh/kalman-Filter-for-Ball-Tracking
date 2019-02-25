#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui.hpp>
#include <opencv/cv.h>
#include<fstream>
#include <pylon/PylonIncludes.h>
#include<math.h>
#include <cmath>
#include <string>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include<time.h>
#include "/home/harsh/kalman extended filter-pendulum/gnuplot_i.hpp"


class ExtendedKalmanFilter
{
    public:
        float** States;
        float** ControlMatrix;
        float** TransitionJacobian;
        float** MeasurementJacobian;
        float** Measurement;
        float** ProcessNoiseCovariance;
        float** MeasurementNoiseCovariance;
        float** Innovation;
        float** ErrorCovariance;
        float** Gain;
        float** function;
        float** temp1;
        float** temp2;
        float** temp3;
        float** temp4;
        float** temp5;
        float** temp6;
        float** z ;
        float** I ;
        float** IDEN;
        bool IsBounce;

        void InitializeFilter();
        void Filter(float *);
};

// Namespace for using pylon objects.
using namespace Pylon;

// Namespace for using GenApi objects
using namespace GenApi;

// Namespace for using opencv objects.
using namespace cv;

// Namespace for using cout.
using namespace std;

#define INTRINSICFILENAME0 "/home/harsh/pylon-calibration/build/results/cam726/intrinsic1.txt"
#define INTRINSICFILENAME1 "/home/harsh/pylon-calibration/build/results/cam728/intrinsic1.txt"

#define DISTORTFILENAME0 "/home/harsh/pylon-calibration/build/results/cam726/distortion_coeffs1.txt"
#define DISTORTFILENAME1  "/home/harsh/pylon-calibration/build/results/cam728/distortion_coeffs1.txt"

#define ROTATIONFILENAME0  "/home/harsh/pylon-calibration/build/results/cam726/rotation1.txt"
#define ROTATIONFILENAME1  "/home/harsh/pylon-calibration/build/results/cam728/rotation1.txt"

#define TRANSLATIONFILENAME0  "/home/harsh/pylon-calibration/build/results/cam726/translation1.txt"
#define TRANSLATIONFILENAME1  "/home/harsh/pylon-calibration/build/results/cam728/translation1.txt"
#define ROWS 480
#define COLS 640

#define DT 0.005
#define DTB 0.01665
#define NUMBEROFSTATES 2
#define NUMBEROFMEASUREMENT 2
#define LENGTH 0.5
#define MASS 0.075
#define Friction 0.05




void mymakematrix(float **&temp, unsigned short int rws, unsigned short int cls) //create a matrix
{
    unsigned short int i;
    temp = new float* [rws];
    for(i=0; i<rws; ++i)
        temp[i] = new float [cls];
}

void mymakevector(float **&temp, unsigned short int rws) //create a matrix
{
    temp = new float* [rws];
}

void myaddmatrix(float **a, float **b, float **sum, unsigned short int rws, unsigned short int cls) //sum = a + b
{
    unsigned short int i, j;
    for (i=0; i<rws; ++i) // till all rows
        for (j=0; j<cls; ++j) //till all columns
            sum[i][j] = a[i][j] + b[i][j]; //add each eleent and store in sum matrix
}

void mysubmatrix(float **a, float **b, float **diff, unsigned short int rws, unsigned short int cls) //diff = a - b
{
    unsigned short int i, j;
    for (i=0; i<rws; ++i) //till all rows
        for (j=0; j<cls; ++j) //till all columns
            diff[i][j] = a[i][j] - b[i][j]; //subtract each element and store in diff matrix
}

void mytransposematrix(float **a, float **b, unsigned short int rws, unsigned short int cls ) //b = a'
{
    unsigned short int i, j;
    for (i=0; i<rws; ++i) //till all rows
        for (j=0; j<cls; ++j) //till all columns
            b[j][i] = a[i][j]; //do transpose
}

void mymulmatrix(float **a, float **b, float **mul, unsigned short int arws, unsigned short int acls, unsigned short int bcls) //b = a'
{
    float sum = 0.0;
    unsigned short int i, j, k;
    for(i=0; i<arws; ++i) //till all rows
    {
        for(j=0; j<bcls; ++j) //till all columns
        {
            sum=0; //sum to zero
            for(k=0; k<acls; ++k) //just one row
                sum = sum + a[i][k]*b[k][j]; //multiplication
            mul[i][j] = sum; //accumulate sum
        }
    }
}

void mymulscalarmatrix(float **a, float** b, float mul, unsigned short int rws, unsigned short int cls) //b = a'
{
    unsigned short int i, j;
    for(i=0; i<rws; ++i) //till all rows
    {
        for(j=0; j<cls; ++j) //till all columns
        {
            b[i][j] = a[i][j]*mul;
        }
    }
}

void myprintmatrix(float **a, unsigned short int rws, unsigned short int cls) //b = a'
{
    unsigned short int i, j;
    for(i=0; i<rws; ++i) //till all rows
    {
        for(j=0; j<cls; ++j) //till all columns
        {
            std::cout<<a[i][j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<std::endl;
}

void mymakeidentitymatrix(float **a, unsigned short int rws, unsigned short int cls) //b = a'
{
    unsigned short int i, j;
    for(i=0; i<rws && i<cls; ++i) //till all rows
    {
        for(j=0; j<cls; ++j)
        {
            if(i==j)
                a[i][j] = 1;
            else
                a[i][j] = 0;
        }
    }
}

void mycholdecmatrix(float** a, float** l, unsigned short int n)
{
    float s;
    unsigned short int i, j, k;
    for(i=0; i<n; ++i)
    {
        for(j=0; j<(i+1); ++j)
        {
            s = 0;
            for (k=0; k<j; ++k)
                s += l[i][k] * l[j][k];
            if(i == j)
                l[i][j] = std::sqrt(a[i][i] - s);
            else
                l[i][j] = (a[i][j] - s)/l[j][j];
        }
    }
}


void ExtendedKalmanFilter::InitializeFilter()
{
    mymakematrix(States, NUMBEROFSTATES, 1);
    mymakematrix(ControlMatrix, NUMBEROFSTATES, 1);
    mymakematrix(TransitionJacobian, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(MeasurementJacobian, 1, NUMBEROFMEASUREMENT);
    mymakematrix(Measurement, NUMBEROFMEASUREMENT, 1);
    mymakematrix(ProcessNoiseCovariance, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(MeasurementNoiseCovariance, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
    mymakematrix(ErrorCovariance, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(Innovation, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
    mymakematrix(Gain, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
    mymakematrix(function, 2, 1);
    mymakematrix(temp1, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(temp2, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(temp3, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
    mymakematrix(temp4, NUMBEROFMEASUREMENT, 1);
    mymakematrix(temp5, NUMBEROFMEASUREMENT, 1);
    mymakematrix(temp6, NUMBEROFMEASUREMENT, 1);
    mymakematrix(I, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT );
    mymakematrix(z, NUMBEROFMEASUREMENT, 1);
    mymakematrix(IDEN, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT );

    IsBounce = false;
    States[0][0] = 1.02;
    States[1][0] = 0;

    IDEN[0][0]= 1;
    IDEN[0][1]= 0;
    IDEN[1][0]=0;
    IDEN[1][1]=1;

    ErrorCovariance[0][0] = 1;
    ErrorCovariance[1][1] = 1;

    ProcessNoiseCovariance[0][0] = 25;
    ProcessNoiseCovariance[1][1] = 25;


    MeasurementNoiseCovariance[0][0] = 1;
    MeasurementNoiseCovariance[1][1] = 1;   //0.5 shifts towards left

    TransitionJacobian[0][0] = 1;
    TransitionJacobian[0][1] = DT;
    TransitionJacobian[1][0] = 0;
    TransitionJacobian[1][1] = 0;

    MeasurementJacobian[0][0] = 0;
    MeasurementJacobian[0][1] = 0;

}

void ExtendedKalmanFilter::Filter(float measurement[])
{
    MeasurementJacobian[0][0] = LENGTH*cos(measurement[2]);
    function[0][1] = measurement[2] + measurement[3]*DT;
    function[1][0] = measurement[3] - (9.8/LENGTH)*(sin(measurement[2]))*DT - (DT*(Friction/MASS)*measurement[3]);
//    cout<<"fass"<<endl;

    States[0][0] = function[0][0] + 0;
     States[1][0] = function[1][0] + 0;
//     cout<<"fass"<<endl;

      TransitionJacobian[1][0] = -(9.8/LENGTH)*(cos(measurement[0]))*DT;
      TransitionJacobian[1][1] = 1 - (Friction/MASS)*DT;


      //-------- P(k) = F(x)*P(k-1)*tr(F(x))+Q(k-1)
      mytransposematrix(TransitionJacobian, temp1, NUMBEROFSTATES, NUMBEROFSTATES);
      mymulmatrix(TransitionJacobian, ErrorCovariance, temp2, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
      mymulmatrix(temp2, temp1, temp3, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
      myaddmatrix(temp3, ProcessNoiseCovariance, ErrorCovariance, NUMBEROFSTATES, NUMBEROFSTATES);
       //-------- P(k) = F(x)*P(k-1)*tr(F(x))+Q(k-1)



      //------------S(k) = H(x)*P(k)*tr(H(x))+R(k)
      mytransposematrix(MeasurementJacobian, temp1, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
      mymulmatrix(ErrorCovariance, temp1, temp2, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
      mymulmatrix(MeasurementJacobian, temp2, Innovation, NUMBEROFMEASUREMENT, NUMBEROFSTATES, NUMBEROFSTATES);
      myaddmatrix(Innovation, MeasurementNoiseCovariance, Innovation, NUMBEROFSTATES, NUMBEROFSTATES);
      //------------S(k) = H(x)*P(k)*tr(H(x))+R(k)


      //--------INVERSE MATRIX CONVERSION STARTS--------------
      int i,j,k;
      float temp;
      float **a;
      float **b;
      float **I;
     int rws = NUMBEROFMEASUREMENT;
      a = Innovation;
      b = temp3;
      I = temp2;
      for(i=0; i<rws; ++i)
          for(j=0; j< rws; ++j)
              b[i][j] = a[i][j];

      for(i=0; i<rws; ++i)
      {
          for(j=0; j< rws; ++j)
          {
              if(i==j)
                  I[i][j]=1;
              else
              {
              mytransposematrix(MeasurementJacobian, temp1, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
              mymulmatrix(ErrorCovariance, temp1, temp2, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
              mymulmatrix(MeasurementJacobian, temp2, Innovation, NUMBEROFMEASUREMENT, NUMBEROFSTATES, NUMBEROFSTATES);
              myaddmatrix(Innovation, MeasurementNoiseCovariance, Innovation, NUMBEROFSTATES, NUMBEROFSTATES);
  }
                  I[i][j]=0;
          }
      }

      for(k=0; k<rws; k++)
      {
          temp=b[k][k];
          for(j=0;j<rws;j++)
          {
              b[k][j]/=temp;
              I[k][j]/=temp;
          }
          for(i=0;i<rws;i++)
          {
              temp=b[i][k];
              for(j=0;j<rws;j++)
              {
                  if(i==k)
                 {
                  mytransposematrix(MeasurementJacobian, temp1, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
                  mymulmatrix(ErrorCovariance, temp1, temp2, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
                  mymulmatrix(MeasurementJacobian, temp2, Innovation, NUMBEROFMEASUREMENT, NUMBEROFSTATES, NUMBEROFSTATES);
                  myaddmatrix(Innovation, MeasurementNoiseCovariance, Innovation, NUMBEROFSTATES, NUMBEROFSTATES);
  }
                      break;
                  b[i][j] -= b[k][j]*temp;
                  I[i][j] -= I[k][j]*temp;
              }
          }
      }

      temp2=I;
      //--------INVERSE MATRIX CONVERSION ENDS--------------


      //----------------KALMAN GAIN-------------
      mytransposematrix(MeasurementJacobian, temp1, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
      mymulmatrix(ErrorCovariance, temp1, temp3, NUMBEROFSTATES, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
      mymulmatrix(temp3, temp2 , Gain, NUMBEROFMEASUREMENT, NUMBEROFSTATES, NUMBEROFSTATES);
      //------------------KALMAN GAIN----------------


      //-----------Y = LENGTH*CURRENT MEASUREMENT - LENGHT*PREVIOUSMEASUREMENT
     z[0][0] = LENGTH*sin(measurement[0]) - LENGTH*sin(measurement[2] + measurement[3]*DT);
      temp4[0][0] = z[0][0];
      temp4[1][0] = 0;
      //------------------------------------------------------------------------

      //---------------PREDICTED STATES = PREVIOUS STATES + GAIN*Y
       mymulmatrix(Gain, temp4, temp5, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
       myaddmatrix(States, temp5, States, NUMBEROFSTATES, NUMBEROFSTATES);
      //---------------------------------------------------------------------

       //---------------ERROR COVARIANCE = (IDEN - GAIN*MEASUREMENT JACOBIAN)*PREVIOUS ERROR COVARIANCE-------
       mymulmatrix(Gain, MeasurementJacobian, temp5, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
       mysubmatrix(IDEN, temp5, temp6, NUMBEROFSTATES, NUMBEROFSTATES);
       mymulmatrix(temp6, ErrorCovariance, ErrorCovariance, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
      //------------------------------------------------------------------------------------
    }

int main( int argc, char** argv )
{
	unsigned long long int i, j;

    Pylon::PylonAutoInitTerm autoInitTerm;
          PylonInitialize();
          static const size_t c_maxCamerasToUse = 2;
         try
         {
              CTlFactory& tlFactory = CTlFactory::GetInstance();
              // Get all attached devices and exit application if no device is found.
              DeviceInfoList_t devices;
              if ( tlFactory.EnumerateDevices(devices) == 0 )
              {
                  throw RUNTIME_EXCEPTION( "No camera present.");
              }
              // Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
              CInstantCameraArray cam( min( devices.size(), c_maxCamerasToUse));
              // Create and attach all Pylon Devices.
              for ( size_t i = 0; i < cam.GetSize(); ++i)
              {
                  cam[ i ].Attach( tlFactory.CreateDevice( devices[ i ]));
                  // Print the model name of the camera.
                  cout << "Using device " << cam[ i ].GetDeviceInfo().GetModelName() << endl;
              }

          cam[0].MaxNumBuffer = 10;
          cam[1].MaxNumBuffer = 10;

      // create pylon image format converter and pylon image
      CImageFormatConverter formatConverter1;
      formatConverter1.OutputPixelFormat= PixelType_BGR8packed;
      CPylonImage pylonImage1;
      CImageFormatConverter formatConverter2;
      formatConverter2.OutputPixelFormat= PixelType_BGR8packed;
      CPylonImage pylonImage2;

          // Start the grabbing of c_countOfImagesToGrab images.
          // The camera device is parameterized with a default configuration which
          // sets up free-running continuous acquisition.
          cam[0].StartGrabbing(GrabStrategy_LatestImageOnly);
          cam[1].StartGrabbing(GrabStrategy_LatestImageOnly);

	cv::Mat img0, img1;
    cv::Mat imghsv0, imghsv1; // Input HSV image : Global cause of trackbar
	cv::Mat imgbin0, imgbin1; // Input Binary image : Filtered object is ball : Global cause of trackbar
	cv::Mat intrinsicMatrix0, intrinsicMatrix1;
	cv::Mat distortionCoefficient0, distortionCoefficient1;
	cv::Mat fundamentalMatrix0, fundamentalMatrix1;

	cv::Mat imgG0, imgBG0;
	cv::Mat imgG1, imgBG1;


//	cv::Mat pt0cv(1, 1, CV_64FC2);
//	cv::Mat pt1cv(1, 1, CV_64FC2);
//    cv::Mat pt2cv(1, 1, CV_64FC2);
//    cv::Mat pt3cv(1, 1, CV_64FC2);  //THESE POINTS CAN BE USED FOR TRIANGULATION
//    cv::Mat pt0Ucv(1, 1, CV_64FC2);//FOR BOTH THE CAMERAS
//	cv::Mat pt1Ucv(1, 1, CV_64FC2);
//    cv::Mat pt2Ucv(1, 1, CV_64FC2);
//    cv::Mat pt3Ucv(1, 1, CV_64FC2);
//	cv::Mat Pt3d(4, 1, CV_64FC1);
//    cv::Mat Pt4d(4, 1, CV_64FC1);

    int filter0[6] = {255, 6, 255, 9, 126, 70}; // Arrey for HSV filter, [Hmax, Hmin, Smax, Smin, Vmax, Vmin]
    int filter1[6] = {255, 6, 255, 9, 126, 70}; // Arrey for HSV filter, [Hmax, Hmin, Smax, Smin, Vmax, Vmin]

	std::fstream file, file2;
	char line[100];

	file.open(INTRINSICFILENAME0, std::fstream::in);
	if(!file.is_open())
	{
		std::cout<<" [ERR] Intrinsic1 Parameter file not found"<<std::endl;
		std::exit(0);
	}
	else
	{
		intrinsicMatrix0.create(3, 3, CV_64FC1);
		for(i=0; i<intrinsicMatrix0.rows; ++i)
		{
			for(j=0; j<intrinsicMatrix0.cols; ++j)
			{
				file>>line;
				intrinsicMatrix0.at<double>(i,j) = std::atof(line);
			}
		}
		file.close();
    }
	std::cout<<intrinsicMatrix0<<std::endl<<std::endl;

	file.open(INTRINSICFILENAME1, std::fstream::in);
	if(!file.is_open())
	{
		std::cout<<" [ERR] Intrinsic1 Parameter file not found"<<std::endl;
		std::exit(0);
	}
    else
	{
		intrinsicMatrix1.create(3, 3, CV_64FC1);
		for(i=0; i<intrinsicMatrix1.rows; ++i)
		{
			for(j=0; j<intrinsicMatrix1.cols; ++j)
			{
				file>>line;
				intrinsicMatrix1.at<double>(i,j) = std::atof(line);
			}
		}
        file.close();
	}
	std::cout<<intrinsicMatrix1<<std::endl<<std::endl;

	file.open(DISTORTFILENAME0, std::fstream::in);
	if(!file.is_open())
	{
		std::cout<<" [ERR] Distortion1 Parameter file not found"<<std::endl;
		std::exit(0);
	}
	else
	{
		distortionCoefficient0.create(5, 1, CV_64FC1);
		for(i=0; i<distortionCoefficient0.rows; ++i)
		{
			for(j=0; j<distortionCoefficient0.cols; ++j)
			{
				file>>line;
				distortionCoefficient0.at<double>(i,j) = std::atof(line);
			}
			//k1[i] = std::atof(line);
		}
		file.close();
	}
	//std::cout<<distortionCoefficient0<<std::endl<<std::endl;

	file.open(DISTORTFILENAME1, std::fstream::in);
	if(!file.is_open())
	{
		std::cout<<" [ERR] Distortion1 Parameter file not found"<<std::endl;
		std::exit(0);
	}
	else
	{
		distortionCoefficient1.create(5, 1, CV_64FC1);
		for(i=0; i<distortionCoefficient1.rows; ++i)
		{
            for(j=0; j<distortionCoefficient1.cols; ++j)
			{
				file>>line;
				distortionCoefficient1.at<double>(i,j) = std::atof(line);
			}
			//k1[i] = std::atof(line);
        }
		file.close();
	}
	//std::cout<<distortionCoefficient1<<std::endl<<std::endl;


	file.open(ROTATIONFILENAME0, std::fstream::in);
	file2.open(TRANSLATIONFILENAME0, std::fstream::in);
	if(!file.is_open() || !file2.is_open())
	{
		std::cout<<" [ERR] fun1 Parameter file not found"<<std::endl;
		std::exit(0);
	}
	else
	{
		fundamentalMatrix0.create(3, 4, CV_64FC1);
		for(i=0; i<fundamentalMatrix0.rows; ++i)
		{
			for(j=0; j<fundamentalMatrix0.cols-1; ++j)
			{
				file>>line;
				fundamentalMatrix0.at<double>(i,j) = std::atof(line);
            }
			file2>>line;
			fundamentalMatrix0.at<double>(i,j) = std::atof(line);
		}
		file.close();
		file2.close();
	}
	//std::cout<<fundamentalMatrix0<<std::endl<<std::endl;

	file.open(ROTATIONFILENAME1, std::fstream::in);
	file2.open(TRANSLATIONFILENAME1, std::fstream::in);
	if(!file.is_open() || !file2.is_open())
	{
		std::cout<<" [ERR] fun1 Parameter file not found"<<std::endl;
		std::exit(0);
	}
	else
	{
		fundamentalMatrix1.create(3, 4, CV_64FC1);
		for(i=0; i<fundamentalMatrix1.rows; ++i)
		{
			for(j=0; j<fundamentalMatrix1.cols-1; ++j)
			{
				file>>line;
				fundamentalMatrix1.at<double>(i,j) = std::atof(line);
			}
			file2>>line;
			fundamentalMatrix1.at<double>(i,j) = std::atof(line);
		}
		file.close();
		file2.close();
	}
	// std::cout<<fundamentalMatrix0<<std::endl<<std::endl;
	//std::cout<<fundamentalMatrix1<<std::endl<<std::endl;

		file.open("3d.txt", std::fstream::out);
		//file2.open("time.txt", std::fstream::out);



    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS,cv::Size(3,3),cv::Point(-1,-1));
	char key = 'r'; // Key press to stop execution
    using namespace std;

    unsigned char status = 0;
	key = 'r';
    double x, y;
    vector<double> xdata, ydata, xreal, yreal;
    int i = 0;
    while(i < 2000)
	{
      CGrabResultPtr ptrGrabResult1;
      CGrabResultPtr ptrGrabResult2;

        // Wait for an image and then retrieve it. A timeout of 5000 ms is used.
      cam[0].RetrieveResult(50, ptrGrabResult1, TimeoutHandling_ThrowException);
      cam[1].RetrieveResult(50, ptrGrabResult2, TimeoutHandling_ThrowException);

      const uint8_t *pImageBuffer1 = (uint8_t *) ptrGrabResult1->GetBuffer();
      const uint8_t *pImageBuffer2 = (uint8_t *) ptrGrabResult2->GetBuffer();

      // Convert the grabbed buffer to pylon imag
      formatConverter1.Convert(pylonImage1, ptrGrabResult1);
      formatConverter2.Convert(pylonImage2, ptrGrabResult2);

      // Create an OpenCV image out of pylon image
      img0= cv::Mat(ptrGrabResult1->GetHeight(), ptrGrabResult1->GetWidth(), CV_8UC3, (uint8_t *) pylonImage1.GetBuffer());
      img1= cv::Mat(ptrGrabResult2->GetHeight(), ptrGrabResult2->GetWidth(), CV_8UC3, (uint8_t *) pylonImage2.GetBuffer());
       

      cv::cvtColor(img0, imghsv0, CV_BGR2HSV); // Convert colour to HSV
      cv::cvtColor(img1, imghsv1, CV_BGR2HSV); // Convert colour to HSV

      cv::inRange(imghsv0, cv::Scalar(filter0[1], filter0[3], filter0[5]), cv::Scalar(filter0[0], filter0[2], filter0[4]), imgG0);
      cv::inRange(imghsv1, cv::Scalar(filter1[1], filter1[3], filter1[5]), cv::Scalar(filter1[0], filter1[2], filter1[4]), imgG1);

      cv::erode(imgG0,imgG0,kernel,cv::Point(-1,-1),2);
      cv::dilate(imgG0,imgG0,kernel,cv::Point(-1,-1),5);
      cv::erode(imgG0,imgG0,kernel,cv::Point(-1,-1),3);

      cv::erode(imgG1,imgG1,kernel,cv::Point(-1,-1),2);
      cv::dilate(imgG1,imgG1,kernel,cv::Point(-1,-1),5);
      cv::erode(imgG1,imgG1,kernel,cv::Point(-1,-1),3);

//         Find all contours
      std::vector<std::vector<cv::Point> > contours1;
      cv::findContours(imgG0.clone(), contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
      std::vector<std::vector<cv::Point> > contours2;
      cv::findContours(imgG1.clone(), contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
//             Fill holes in each contour
      cv::drawContours(imgG0, contours1, -1, CV_RGB(255, 255, 255), 3);
//                cout << contours1.size()<<endl;
      cv::drawContours(imgG1, contours2, -1, CV_RGB(255, 255, 255), 3);
//                cout << contours2.size();

                if(contours1.size()>=1 && contours2.size()>=1)
                {
                    double avg_x1(0), avg_y1(0); // average of contour points
                     double avg_x2(0), avg_y2(0); // average of contour points
                     double avg_x3(0), avg_y3(0); // average of contour points
                     double avg_x4(0), avg_y4(0); // average of contour points

                    for (int j = 0; j < contours1[0].size(); ++j)
                    {
                       avg_x1 += contours1[0][j].x;
                       avg_y1 += contours1[0][j].y;

                    }
                    for (int j = 0; j < contours1[1].size(); ++j)
                    {
                       avg_x3 += contours1[1][j].x;
                       avg_y3 += contours1[1][j].y;

                    }

                    avg_x1 /= contours1[0].size();
                    avg_y1 /= contours1[0].size();
                    avg_x3 /= contours1[1].size();
                    avg_y3 /= contours1[1].size();
                     double l = sqrt((avg_x1-avg_x3)*(avg_x1-avg_x3)+(avg_y1-avg_y3)*(avg_y1-avg_y3));
                     cout<<l<<endl;
//                    pt0cv.at<cv::Vec2d>(0, 0)[0] = int(avg_x1);// = int(488);
//                    pt0cv.at<cv::Vec2d>(0, 0)[1] = int(avg_y1);// = 149;
//                    pt2cv.at<cv::Vec2d>(0, 0)[0] = int(avg_x3);// = int(488);
//                    pt2cv.at<cv::Vec2d>(0, 0)[1] = int(avg_y3);// = int(488);

//                    cout << avg_x1 << " " << avg_y1 << endl;
                    cv::circle( img0, cv::Point(avg_x1, avg_y1), 6, cv::Scalar(0, 0, 255), 8, 0 );
                    cv::circle( img0, cv::Point(avg_x3, avg_y3), 6, cv::Scalar(255, 0, 0), 8, 0);


                   for (int j = 0; j < contours2[0].size(); ++j)
                       {

                                   avg_x2 += contours2[0][j].x;
                                   avg_y2 += contours2[0][j].y;

                       }
                   for (int j = 0; j < contours2[1].size(); ++j)
                       {

                                   avg_x4 += contours2[1][j].x;
                                   avg_y4 += contours2[1][j].y;

                       }

                                avg_x2 /= contours2[0].size();
                                avg_y2 /= contours2[0].size();
                                avg_x4 /= contours2[1].size();
                                avg_y4 /= contours2[1].size();
//                                pt1cv.at<cv::Vec2d>(0, 0)[0] = int(avg_x2);// = int(488);
//                                pt1cv.at<cv::Vec2d>(0, 0)[1] = int(avg_y2);// = 149;
//                                pt3cv.at<cv::Vec2d>(0, 0)[0] = int(avg_x4);// = int(488);
//                                pt3cv.at<cv::Vec2d>(0, 0)[0] = int(avg_x4);// = int(488);

//                                cout << avg_x2 << " " << avg_y2 << endl;
                                cv::circle( img1, cv::Point(avg_x2, avg_y2), 2, cv::Scalar(0, 0, 255), 8, 0 );
                                cv::circle( img1, cv::Point(avg_x4, avg_y4), 2, cv::Scalar(255, 0, 0), 8, 0 );

//                                cv::undistortPoints(pt0cv, pt0Ucv, intrinsicMatrix0, distortionCoefficient0);
//                                cv::undistortPoints(pt1cv, pt1Ucv, intrinsicMatrix1, distortionCoefficient1);
//                                cv::undistortPoints(pt2cv, pt2Ucv, intrinsicMatrix0, distortionCoefficient0);
//                                cv::undistortPoints(pt3cv, pt3Ucv, intrinsicMatrix1, distortionCoefficient1);
//                                cout<<"yahan tak cahl gya"<<endl;
                                float thetak;
                                float y;
                                float prevvel;
                                thetak = (atan((avg_x3 - avg_x1)/(avg_y3 - avg_y1)));
                                y = thetak *(180/3.1428);
                                cout<<"measured angle :-"<<y<<endl;
                               //  cout <<thetak<<endl;

   // ------------------------------------------KALMAN FILTER------------------------------------

                                ExtendedKalmanFilter EKF;
                                EKF.InitializeFilter();                                       
                                cout<<"fass"<<endl;

                                float currpos[1], prevpos[1], measurement[4];

                                 currpos[0] = thetak;

                             switch(status)
                                {

                                   case 0:
                                    EKF.States[0][0] = currpos[0];
                                    EKF.States[1][0] = 0;
                                    prevpos[0] = currpos[0];
//                                  EKF.States[2][0] = currpos[2];
//                                  prevpos[1] = currpos[1];
//                                  prevpos[2] = currpos[2];
                                    status++;
                                    break;

                                    case 1:
                                    EKF.States[0][0] = currpos[0];
                                    EKF.States[1][0] = (currpos[0]-prevpos[0])/DT;
                                    prevvel =  EKF.States[1][0];
//                                  EKF.States[2][0] = currpos[2];
//                                  EKF.States[3][0] = (currpos[0]-prevpos[0])/DT;
//                                  EKF.States[4][0] = (currpos[1]-prevpos[1])/DT;
//                                  EKF.States[5][0] = (currpos[2]-prevpos[2])/DT;
                                    measurement[2]=prevpos[0];
                                    measurement[3]=0;
                                    prevpos[0] = currpos[0];

//                                  prevpos[1] = currpos[1];
//                                  prevpos[2] = currpos[2];
                                    status++;
                                    break;

                                    case 2:
                                    measurement[0] = currpos[0]; //new pos
                                    measurement[1] = (currpos[0] - prevpos[0])/DT;  //new velocity
                                    measurement[2]=prevpos[0];   //purani pos
                                    measurement[3]=prevvel;    //purani velocity
                                    prevvel=measurement[1];

//                                  measurement[5] = (currpos[2] - prevpos[2])/DTB;
//                                  measurement[0] = currpos[0];
//                                  measurement[1] = currpos[1];
//                                  measurement[2] = currpos[2];

                                     EKF.Filter(measurement);
//                                   std::cout<<currpos[0]<<" "<<(currpos[0] - prevpos[0])/DT<<std::endl;
                                     std::cout<<EKF.States[0][0]<<" "<<EKF.States[1][0]<<std::endl;
                                     std::cout<<std::endl;
                                     prevpos[0] = currpos[0];
//                                   prevpos[1] = currpos[1];
//                                   prevpos[2] = currpos[2];
//                                   cout<<avg_x1<<endl;
                                     xdata.push_back(EKF.States[0][0]);
                                     ydata.push_back(i);
                                     yreal.push_back(i);
                                     xreal.push_back(measurement[0]);
                                     float s = EKF.States[0][0]*(180/3.1428);
                                     cout<<"measured state:-"<<s<<endl;

                                     double finalx1 = avg_x3 +(int) l*sin(EKF.States[0][0]);
                                     double finaly1 = avg_y3 + (int )l*cos(EKF.States[0][0]);

                                     cv::circle( img0, cv::Point(finalx1, finaly1), 2, cv::Scalar(0, 255, 0), 8, 0);

                                     break;


                                        }

                                }
//-------------THIS IS TO CONVERT THE PIXEL CO-ORDINATES INTO 3D CO-ORDINATES-------------------------------------------------------------------
//			cv::undistortPoints(pt0cv, pt0Ucv, intrinsicMatrix0, distortionCoefficient0);
//			cv::undistortPoints(pt1cv, pt1Ucv, intrinsicMatrix1, distortionCoefficient1);
//          cv::undistortPoints(pt2cv, pt2Ucv, intrinsicMatrix0, distortionCoefficient0);
//          cv::undistortPoints(pt3cv, pt3Ucv, intrinsicMatrix1, distortionCoefficient1);
//          std::cout<<" "<<pt0Ucv.at<cv::Vec2f>(0,0)[0]<<" "<<pt0Ucv.at<cv::Vec2f>(0,0)[1]<<std::endl;
//          std::cout<<" "<<pt1Ucv.at<cv::Vec2f>(0,0)[0]<<" "<<pt1Ucv.at<cv::Vec2f>(0,0)[1]<<std::endl;
//          cv::triangulatePoints(fundamentalMatrix0, fundamentalMatrix1, pt0Ucv, pt1Ucv, Pt3d);
//          cv::triangulatePoints(fundamentalMatrix0, fundamentalMatrix1, pt2Ucv, pt3Ucv, Pt4d);
//          std::cout<<Pt3d.at<float>(0,0)/Pt3d.at<float>(3,0)<<" "<<Pt3d.at<float>(1,0)/Pt3d.at<float>(3,0)<<" "<<Pt3d.at<float>(2,0)/Pt3d.at<float>(3,0)<<" "<<std::endl;
//          std::cout<<Pt4d.at<float>(0,0)/Pt4d.at<float>(3,0)<<" "<<Pt4d.at<float>(1,0)/Pt4d.at<float>(3,0)<<" "<<Pt4d.at<float>(2,0)/Pt4d.at<float>(3,0)<<" "<<std::endl;
//			file<<Pt3d.at<double>(0,0)/Pt3d.at<double>(3,0)<<" "<<Pt3d.at<double>(1,0)/Pt3d.at<double>(3,0)<<" "<<Pt3d.at<double>(2,0)/Pt3d.at<double>(3,0)<<"\n";
//          file<<Pt4d.at<double>(0,0)/Pt4d.at<double>(3,0)<<" "<<Pt4d.at<double>(1,0)/Pt4d.at<double>(3,0)<<" "<<Pt4d.at<double>(2,0)/Pt4d.at<double>(3,0)<<"\n";
//---------------------------------------------------------------------------------------------------------------------------------
                                        namedWindow("cam726",WINDOW_FREERATIO);
//                                        namedWindow("cam728",WINDOW_FREERATIO);
                                        namedWindow("cam726_BIN",WINDOW_FREERATIO);
//                                        namedWindow("cam728_BIN",WINDOW_FREERATIO);
                                        cv::imshow("cam726", img0); // Show centroid image
//                                        cv::imshow("cam728", img1); // Show centroid image
                                        cv::imshow("cam726_BIN",imgG1); // Show filtered image
//                                        cv::imshow("cam728_BIN",imgG0); // Show filtered image
                                        cv::resizeWindow("cam726", 200,200);
                                        cv::resizeWindow("cam728", 200,200);
                                        cv::resizeWindow("cam726_BIN", 200,200);
                                        cv::resizeWindow("cam728_BIN", 200,200);
                                        key = cv::waitKey(1); // Wait for 1ms for intrupt from user to exit
                                        if(key == 'q') // If user presses 'q'
                                            break; // Exit while loop
                                        cout<<i<<endl;
                                        i++;

                                    }
                                         try
                                        {
                                        // Create Gnuplot object

                                        Gnuplot gp;

                                        // Configure the plot

                                        gp.set_style("lines");
                                        gp.set_xlabel("x");
                                        gp.set_ylabel("y");


                                        // Plot the data

                                        gp.plot_xy(ydata, xdata);
                                        gp.plot_xy(yreal, xreal);
                                        cout << "Press Enter to quit...";
                                        cin.get();

                                        return 0;
                                      }
                                      catch (const GnuplotException& error) {

                                        cerr << error.what() << endl;
                                        return 1;
                                    }
                                        waitKey();

                                        file.close();
                                        file2.close();

                                        cv::destroyAllWindows(); // Distroid all display windows
                                        img0.release();
                                        img1.release();
                                        imghsv0.release();
                                        imghsv1.release();
                                        imgG0.release();
                                        imgG1.release();

                                        return 0;
                                          }

                                        catch (GenICam::GenericException &e)
                                        {
                                            // Error handling.
                                            cerr << "An exception occurred." << endl
                                            << e.GetDescription() << endl;
                                    //            exitCode = 1;
                                        }

                                    //        return exitCode;
                                        PylonTerminate();

                                }


