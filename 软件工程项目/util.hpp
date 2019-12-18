

#pragma once

#include "custom_definations.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <mutex>
using namespace std;
using namespace cv;

#define FOREACH(i,arrSize) for(int i=0;i<arrSize;i++)
#define FORPAIR(i,j,arrSize) for(int i=0;i<arrSize;i++)for(int j=i+1;j<arrSize;j++)
#define FORPAIR_ORDERED(i,j,arrSize) for(int i=0;i<arrSize;i++)for(int j=0;j<arrSize;j++)


// 弧度转角度的比例系数，弧度指乘上Rad2Deg可以得到角度
#define Rad2Deg 57.295779513
// 与上面相反
#define Deg2Rad 0.0174532925


// 用TimeStart开始标记时间，TimePoint打印当前与上一次时间标记之间的时间间隔，以毫秒为单位
// 其中caption表示打印的提示内容
#define TIME_START double _clock_start_time=(double)getTickCount(),_clock_cur_time;\
					cout << endl << "Clock Point Started... " << endl;
#define TIME_POINT(caption) _clock_cur_time = (double)getTickCount();\
  					cout << caption << " :" << (_clock_cur_time - _clock_start_time) * 1000.0 / getTickFrequency() << "ms" << endl;\
					_clock_start_time = _clock_cur_time;

#define ACCURATE_TIME_START clock_t _ac_clock_start_time=clock(),_acc_cur_time;\
                            cout << endl << "Clock Point Started..." << endl;
#define ACCURATE_TIME_POINT(caption) _acc_cur_time = clock();\
                            cout << caption << " :" << (_acc_cur_time - _ac_clock_start_time) /1000.0 << endl;\
                            _ac_clock_start_time = _acc_cur_time;

#define LOG(str) cout << #str << endl;

// add an image to display set
#define DEBUG_DISPLAY(img) DebugDisplayManager::AddDisplayImage(#img,img);


template<typename T>
struct _locked_variable
{
private:
    mutex _lock;
public:
    T variable;
    inline void Lock()
    {
        _lock.lock();
    }
    inline void Unlock()
    {
        _lock.unlock();
    }
};

template<typename T>
using Lock = struct _locked_variable<T> ;



// camera image data including image and information related to the image
typedef struct _img_data
{
    cv::Mat image;
    int index;
    // the state at the image time
    Point2f ptzSpeed,ptzAngle,worldPosition;
    int shootSpeed;
    
    void copyTo(struct _img_data &other)
    {
        image.copyTo(other.image);
        image.release();
        other.index = index;
        other.ptzSpeed = ptzSpeed;
        other.ptzAngle = ptzAngle;
        other.worldPosition = worldPosition;
        other.shootSpeed = shootSpeed;
    }
}ImageData;

// describing debug display image
typedef struct _dbg_dpl_img{
    string title;
    Mat image;
    bool visible;

    _dbg_dpl_img(string ttl,Mat img)
    {
        title = ttl;
        image = img; 
        visible = true;
    }

    inline bool IsVisible()
    {
        return !image.empty() && visible;
    }
}DebugDisplayImage;

// 装甲类型定义
typedef int ArmorColorType;
#define ARMOR_RED 2
#define ARMOR_BLUE 0
#define ARMOR_UNKNOWN 1

class ModuleBase
{
public:
    int moduleID;

    ModuleBase(int id)
    {
        moduleID = id;
    }

    virtual void EnableModule() = 0;
    virtual void DisableModule() = 0;
    virtual void Update(ImageData &frame,float dtTime) = 0;
};

class DebugDisplayManager
{
public:
    static vector<DebugDisplayImage> imgs;
    static void AddDisplayImage(string title,Mat img)
    {
        FOREACH(i,imgs.size())if (imgs[i].title == title)
        {
            img.copyTo(imgs[i].image);
            imgs[i].visible = true;
            return ;
        }
        imgs.push_back(DebugDisplayImage(title,img.clone()));
    }
    static void DisplayAll()
    {
        FOREACH(i,imgs.size()) if (imgs[i].IsVisible())
        {
            imshow(imgs[i].title,imgs[i].image);
            imgs[i].visible = false;
        }
    }
private :
    //bool on_using;
};
vector<DebugDisplayImage> DebugDisplayManager::imgs;

void GetHist(Mat grey,Mat &hist)
{
    //定义求直方图的通道数目，从0开始索引  
    int channels[]={0};  
    //定义直方图的在每一维上的大小，例如灰度图直方图的横坐标是图像的灰度值，就一维，bin的个数  
    //如果直方图图像横坐标bin个数为x，纵坐标bin个数为y，则channels[]={1,2}其直方图应该为三维的，Z轴是每个bin上统计的数目  
    const int histSize[]={256};  
    //每一维bin的变化范围  
    const float range[]={0,256};  
    //所有bin的变化范围，个数跟channels应该跟channels一致  
    const float* ranges[]={range};  
    //计算直方图，hist大小为256*1，每行存储的统计的该行对应的灰度值的个数  
    calcHist(&grey,1,channels,Mat(),hist,1,histSize,ranges,true,false);
}

void GetHist(Mat grey,Mat &hist,Mat &hist_graph)
{
    GetHist(grey,hist);
    //找出直方图统计的个数的最大值，用来作为直方图纵坐标的高  
    double maxValue=0;  
    //最大值取整  
    int rows=256;  
    //定义直方图图像，直方图纵坐标的高作为行数，列数为256(灰度值的个数)  
    //因为是直方图的图像，所以以黑白两色为区分，白色为直方图的图像  
    hist_graph=Mat::zeros(rows,256,CV_8UC1);
    //直方图图像表示  
    for(int i=0;i<256;i++)  
    {
        //取每个bin的数目  
        int temp=(int)(hist.at<float>(i,0));
	if (temp > rows )temp = rows;
        //如果bin数目为0，则说明图像上没有该灰度值，则整列为黑色  
        //如果图像上有该灰度值，则将该列对应个数的像素设为白色  
        if(temp)  
        {  
            //由于图像坐标是以左上角为原点，所以要进行变换，使直方图图像以左下角为坐标原点  
            hist_graph.col(i).rowRange(Range(rows-temp,rows))=255;   
        }
    }
}

// Calculate the sum value in range(minThresh,maxThresh) / sum of all for hist graph
float GetHistRangePercentage(Mat grey,int minThresh,int maxThresh)
{
    Mat hist;
    GetHist(grey,hist);
    int sum = 0;
    for(int i=minThresh;i<maxThresh;i++)
        sum += hist.at<float>(i,0);
    return (float)sum / (grey.cols * grey.rows);
}
// make sure roi is in image region. otherwise an error might be caused.
void MakeRectSafe(Mat bound,Rect &roi)
{
    roi.x = roi.x < 0 ? 0 : roi.x;
    roi.y = roi.y < 0 ? 0 : roi.y;
    if (roi.x >= bound.cols) roi.x = bound.cols-1;
    if (roi.y >= bound.rows) roi.y = bound.rows-1;
    if (roi.x + roi.width > bound.cols) roi.width = bound.cols - roi.x;
    if (roi.y + roi.height > bound.rows) roi.height = bound.rows - roi.y;
}

inline float SqrLength(Point p1, Point p2)
{
    return (p1.x - p2.x)*(p1.x -p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

inline float Length(Point p1, Point p2)
{
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}

inline float Length(Point2f v)
{
    return sqrt(v.x*v.x+v.y*v.y);
}

inline float Length(Point3f v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline Point2f GetRectCenterPoint(Rect rect)
{
    return Point2f(rect.x + (float)rect.width / 2,rect.y + (float)rect.height / 2);
}

Point2f CalculateGravityAngle(float shootAngle,float shootSpeed,float distance)
{
    float time = distance / shootSpeed , h = 4.9f * time * time;
    return Point2f(0, atan(h/distance));
}

class PID
{
public:
    double kp,ki,kd,max_out,max_i;

    void SetPIDParams(double _kp,double _ki,double _kd)
    {
        i = lst_err = 0;
        kp = _kp;
        ki = _ki;
        kd = _kd;
    }

    void SetLimitParams(double _mout,double _mi)
    {
        max_out = _mout;
        max_i = _mi;
    }

    double calc(double error)
    {
        i += error * ki;
        float res = kp * error + absmin(i,max_i) + kd * (error - lst_err);
        lst_err = error;
        return absmin(res,max_out);
    }
protected:
    double i,lst_err;

    inline double absmin(double val,double limit)
    {
        return val > 0 ? min(val,limit) : -min(-val,limit);
    }
};
