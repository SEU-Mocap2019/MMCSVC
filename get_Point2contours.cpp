#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


extern "C"
{
Point pointTest( InputArray _contour, Point2f pt, bool measureDist ,double & mindis)
{
    
    Point a;

    double result = 0;
    Mat contour = _contour.getMat();
    int i, total = contour.checkVector(2), counter = 0;
    int depth = contour.depth();
    CV_Assert( total >= 0 && (depth == CV_32S || depth == CV_32F));

    bool is_float = depth == CV_32F;
    double min_dist_num = FLT_MAX, min_dist_denom = 1;
    Point ip(cvRound(pt.x), cvRound(pt.y));

    if( total == 0 )
        {
            cout<<"error";
            mindis=100000;
            return a;
            }

    const Point* cnt = contour.ptr<Point>();
    const Point2f* cntf = (const Point2f*)cnt;


  
        Point2f v0, v;
        Point iv;

        if( is_float )
        {
            v = cntf[total-1];
        }
        else
        {
            v = cnt[total-1];
        }

        

            for( i = 0; i < total; i++ )
            {
                double dx, dy, dx1, dy1, dx2, dy2, dist_num, dist_denom = 1;

                v0 = v;
                if( is_float )
                    v = cntf[i];
                else
                    v = cnt[i];

                dx = v.x - v0.x; dy = v.y - v0.y;
                dx1 = pt.x - v0.x; dy1 = pt.y - v0.y;
                dx2 = pt.x - v.x; dy2 = pt.y - v.y;

                if( dx1*dx + dy1*dy <= 0 )
                    dist_num = dx1*dx1 + dy1*dy1;
                else if( dx2*dx + dy2*dy >= 0 )
                    dist_num = dx2*dx2 + dy2*dy2;
                else
                {
                    dist_num = (dy1*dx - dx1*dy);
                    dist_num *= dist_num;
                    dist_denom = dx*dx + dy*dy;
                }

                if( dist_num*min_dist_denom < min_dist_num*dist_denom )
                {
                    a=cnt[i];

                    min_dist_num = dist_num;
                    min_dist_denom = dist_denom;
                    if( min_dist_num == 0 )
                        break;
                }

                if( (v0.y <= pt.y && v.y <= pt.y) ||
                   (v0.y > pt.y && v.y > pt.y) ||
                   (v0.x < pt.x && v.x < pt.x) )
                    continue;

                dist_num = dy1*dx - dx1*dy;
                if( dy < 0 )
                    dist_num = -dist_num;
                counter += dist_num > 0;
            }

            result = std::sqrt(min_dist_num/min_dist_denom);
            
            if( counter % 2 == 0 )
                result = -result;
        
    
    mindis=result;
    return a;
}





/** @function main */
void test(int*x, int*y,int shape)
{

  Mat src = imread("002err1.jpg");
  Mat src_gray;
  cvtColor( src, src_gray, CV_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );
  /// Create a sequence of points to make a contour:

  /// Draw it in src
vector<Point> test;
  /// Get the contours
  vector<vector<Point> > contours; 
  vector<Vec4i> hierarchy;
  Mat src_copy = src.clone();

  Mat canny_output;
  int thresh=100;
  /// 用Canny算子检测边缘
  Canny( src_gray, canny_output, thresh, thresh*2, 3 );



  findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
Mat allContoursResult(src_copy.size(), CV_8U, cv::Scalar(255));  
  //drawContours(allContoursResult,contours,-1,Scalar(0),2);
  //cout<<contours[0]<<" "<<contours[1]<<endl<<endl;
  //imshow("allContours",allContoursResult);
  //Mat drawing = Mat::zeros(src_copy.size(), CV_8UC3 );
  //drawContours(drawing, contours, 0, (0, 0, 255), hierarchy, 8, Point());
  //imshow("result", drawing);
  //waitKey(0);
Point a;
Point atemp;
double mindistemp;
double mindis=1000000;
    for(int i=0;i<shape;++i)
	{
        for (int j = 0; j < hierarchy.size(); j++)
        {
            atemp=pointTest(contours[j],cv::Point(x[j],y[j]),true,mindistemp);
            if(mindistemp<mindis)
            {
                mindis=mindistemp;
                a=atemp;
            }   
        }
        //a=pointTest(contours[0],cv::Point(x[i],y[i]),true,mindis);
        x[i]=a.x;
        y[i]=a.y;
     }
  /// Calculate the distances to the contour

  /// Depicting the  distances graphically


  /// Create Window and show your results

    
  return;
}
int fun(int *a)
{
    for(int i=0;i<sizeof(a);i++)
    {
        
        a[i]+=10;
    }
    return a[2];
}

}

//g++ -o get_Point2contours.so -shared -fPIC get_Point2contours.cpp -lopencv_core -lopencv_highgui -lopencv_imgcodecs