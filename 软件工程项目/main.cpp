
#include "custom_definations.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include "armorDetector.hpp"
#include "RMVideoCapture.hpp"
#include "serial.hpp"
#include <thread>
#include "camview.hpp"         
#include <unistd.h>
#include "armorTracker.hpp"
#include "armorHiter.hpp"
#include "dfcHiter.hpp"//dfc
#include "dnnManager.hpp"
#include "autolead.hpp"
#include "dsfReco.hpp"
#include "dsfHiter.hpp"
#include "autoTracker.hpp"
using namespace cv;
using namespace std;


void ProcessFullFunction(ImageData &frame);
void ProcessAlgorithmFunction(ImageData &frame);
void ArmorDetectDebug(ImageData &frame);
void DSFDetectDebug(ImageData &frame);
void AutoLeadDebug(ImageData &frame);
void AutoBlockDebug(ImageData &frame);
double lastt = getTickCount(), curt = lastt,checkpoint_time = lastt,dtTime = 0;
int fps  = 0,frames = 0;

// returns delta time
void UpdateFPS()
{
    frames++;
    curt = (double)getTickCount();
    if (curt - checkpoint_time >= getTickFrequency())
    {
        checkpoint_time = curt;
        fps = frames;
        frames = 0;
        cout <<"fps:"<< fps << endl;
    }
    dtTime = (curt - lastt) / getTickFrequency();
    lastt = curt;
}

void DisplayFPS(Mat &m)
{
    char fpsstr[30];
    sprintf(fpsstr,"fps:%d",fps);
    putText(m,fpsstr,Point(20,20),FONT_HERSHEY_SIMPLEX,1.0,Scalar(0,0,0),2);
}

//----global class objects

// camera thread variables

#if defined DEVICE_TX
RMVideoCapture rmcap("/dev/video1",3);
#elif defined DEVICE_MANIFOLD
RMVideoCapture rmcap("/dev/video0",3);
#endif

bool threadContinueFlag = true;
char waitedKey = 0;
bool onpause = false;
Lock<ImageData> frameData;
// modules definations
SerialManager *serial_ptr = NULL;
CameraView *camview_ptr = NULL;
ArmorBaseDetector *armor_detector_ptr = NULL;
ArmorTrackerBase *armor_tracker_ptr = NULL;
ArmorHiter *armor_hiter_ptr = NULL;
DfcHiter *dfc_hiter_ptr = NULL;
DnnManager *dnn_manager_ptr = NULL;
DSFReco *dsf_reco_ptr = NULL;
DSFHiter *dsf_hiter_ptr = NULL;
AutoLeader *autoleader_ptr = NULL;
AutoBlockTracker *autoBlock_ptr = NULL;


void SetCameraExposure(bool autoExp = false,int expTime = 64)
{
    rmcap.setExposureTime(autoExp,expTime);
    rmcap.startStream();
}

void ImageCollectThread()
{
    SerialPort serialPort(ConfigurationVariables::GetInt("Port",1));
    SerialManager serialManager(&serialPort);
    serial_ptr = &serialManager;

    int frameIndex = 0;
    Mat img;
    while(threadContinueFlag)
    {
        try{
            // Get Image
            rmcap >> img;
            if (ConfigurationVariables::resolutionType == 0)
                resize(img,img,Size(960,720));
            // 设置当前帧的电控参数
            serialManager.UpdateReadData();
            frameData.Lock();
            //{
                frameData.variable.ptzSpeed = ElectronicControlParams::PTZSpeed;
                frameData.variable.ptzAngle = ElectronicControlParams::PTZAngle;
                frameData.variable.worldPosition = ElectronicControlParams::worldPosition;
                frameData.variable.shootSpeed = ElectronicControlParams::shotSpeed;
                // 告知处理线程，图像准备完成
                frameData.variable.image = img;
                frameData.variable.index = frameIndex++;
            //}
            frameData.Unlock();
            if (frameIndex == 1)
                SetCameraExposure(ConfigurationVariables::GetBool("autoExposure",false),ConfigurationVariables::GetInt("exposureTime",64));

        }
        catch(...)
        {
            cout << "Error in Collect." << endl;
        }
    }
}

void ImageProcessThread()
{

    // 初始化各模块
    char camparams[100] ;
    sprintf(camparams,FILEDIR(camparams_%d.xml),ConfigurationVariables::resolutionType);
    CameraView camview(camparams);
    camview_ptr = &camview;

    PolyMatchArmorDetector armor_detector;
    armor_detector_ptr = (ArmorBaseDetector*)(&armor_detector);
    armor_detector_ptr->SetTargetArmor(2 - ElectronicControlParams::teamInfo);

    LinearPredictor linear_predictor; 

    DnnManager dnnManager(FILEDIR(model/mnist.cfg), FILEDIR(model/mnist.weights), FILEDIR(model/fire.cfg),
                          FILEDIR(model/fire.weights), FILEDIR(model/armor.cfg), FILEDIR(model/armor.weights)); // file names
    dnn_manager_ptr = &dnnManager;

    PredictPIDArmorTracker predict_pid_tracker(camview_ptr,armor_detector_ptr,dnn_manager_ptr,&linear_predictor);
    armor_tracker_ptr = &predict_pid_tracker;

    // 等待串口类被初始化
    while(!serial_ptr);

#if defined ROBOT_SENTINEL
    SentinelArmorHiter sentinelHiter(serial_ptr,armor_tracker_ptr);
    armor_hiter_ptr = &sentinelHiter;
#elif defined ROBOT_INFANCY
    InfancyArmorHiter infancyHiter(serial_ptr,armor_tracker_ptr);
    armor_hiter_ptr = &infancyHiter;
#endif
    // 神符打击和自动引导模块的定义..定制需求添加在这里
    DSFReco dsfReco(camview_ptr,dnn_manager_ptr); 
    dsf_reco_ptr = &dsfReco;

    DSFHiter dsfHiter(serial_ptr,dsf_reco_ptr);
    dsf_hiter_ptr = &dsfHiter;

    DfcHiter dfcHiter( serial_ptr, camview_ptr);
    dfc_hiter_ptr = &dfcHiter;
     
    AutoLeader autoleader(serial_ptr,camview_ptr);
    autoleader_ptr = &autoleader;

    AutoBlockTracker autoBlock(serial_ptr);
    autoBlock_ptr = &autoBlock;

    //初始化serialManager中的模块列表
    serial_ptr->RegisterModule(armor_hiter_ptr);
    serial_ptr->RegisterModule(dfc_hiter_ptr);
    serial_ptr->RegisterModule(autoleader_ptr);

    int lastIndex = 0;
    ImageData processFrame;
    while(threadContinueFlag)
    {
        if (waitedKey == 'p')
        {
            waitedKey = 0;
            onpause = !onpause;
        }
        if (onpause) continue;
        try
        {
            // 当前图像获取线程还不能提供最新图像
            if (lastIndex >= frameData.variable.index) continue;
//LOG(A)
            // 拷贝获取线程得到的图像，防止内存错误
            frameData.Lock();
                frameData.variable.copyTo(processFrame);
            frameData.Unlock();
//LOG(B)
            lastIndex = processFrame.index;
            if (processFrame.image.empty()) break;
            UpdateFPS();
            // 处理该帧数据
            if (ConfigurationVariables::MainEntry < 4)
                ProcessFullFunction(processFrame);
            else 
                ProcessAlgorithmFunction(processFrame);

//LOG(C)
            DisplayFPS(processFrame.image);
            if(DEBUG_MODE)
                DEBUG_DISPLAY(processFrame.image);
            waitedKey = 0;

        }
        catch(...)
        {
            cout << "Error in process." << endl;
        }
        // 串口发送当前帧数据
        serial_ptr->FlushData();
//LOG(D)
        // 每秒更新一次configuration
        if(frames == 0 && ConfigurationVariables::KeepUpdateConfiguration)
        {
            // reloading configuration variables
            ConfigurationVariables::ReadConfiguration(false);
            if (!ConfigurationVariables::Loaded) cout << "Error when update configuration variables." << endl;
        }
    }
}

void ImageDisplayThread()
{
    while(threadContinueFlag)
    {
        if (ConfigurationVariables::DebugMode)
        {
            try{
                if (!onpause || DebugDisplayManager::imgs.size())
                    DebugDisplayManager::DisplayAll();
                int c = waitKey(1);
                if (c != -1) waitedKey = (char)c;
                if (waitedKey == 'x') threadContinueFlag = false;
            }catch(...){
                cout << "Error in Display." << endl;
            }
        }
	else
	    sleep(1);
    }
}

int main() {
    // init configuration
    ConfigurationVariables::ReadConfiguration(true);
    if (!ConfigurationVariables::Loaded)
        cout << "Load Configuration Failed. Using default values." << endl;
    else
        cout << "Configuration Loaded Successfully." << endl;
    ElectronicControlParams::teamInfo = ConfigurationVariables::GetInt("StartArmorType",1);
    // init camera capture
    rmcap.setVideoFormat(ConfigurationVariables::resWidth,ConfigurationVariables::resHeight,ConfigurationVariables::resolutionType != 0);//
    bool autoexp = ConfigurationVariables::GetBool("autoExposure",false);
    rmcap.setExposureTime(autoexp,ConfigurationVariables::GetInt("exposureTime",64));
    //rmcap.setExposureTime(false,10);
    rmcap.startStream();
    // calibration process differs
    if (ConfigurationVariables::MainEntry == 10)
    {
        char camparams[100] ;
        sprintf(camparams,FILEDIR(camparams_%d.xml),ConfigurationVariables::resolutionType);
        CameraView::CalibrateCameraProcess(camparams,rmcap);
        return 0;
    }
    // init modules
    // init threads

    thread proc_thread(ImageProcessThread);
    thread display_thread(ImageDisplayThread);
    thread collect_thread(ImageCollectThread);

    collect_thread.join();
    proc_thread.join();
    display_thread.join();

    return 0;
}

void ProcessFullFunction(ImageData &frame)
{
    // 调试模块的模式下 强制打开模块
    switch(ConfigurationVariables::MainEntry)
    {
    case 1: serial_ptr->EnableModule(2);break;
    case 2: serial_ptr->EnableModule(1);break;
    case 3: serial_ptr->EnableModule(4);break;
    }
    serial_ptr->UpdateCurModule(frame,dtTime);
}

void ProcessAlgorithmFunction(ImageData &frame)
{
    switch(ConfigurationVariables::MainEntry){
        case 4:  // armor detect
            ArmorDetectDebug(frame);
            break;
        case 5:
            DSFDetectDebug(frame);
            break;
        case 6:
            AutoLeadDebug(frame);
            break;
        case 7:
            AutoBlockDebug(frame);
        break;
    }
}

void ArmorDetectDebug(ImageData &frame)
{
    // test detector:
    //armor_detector_ptr->DetectArmors(frame.image);
    // test tracker
    Point2f res = armor_tracker_ptr->UpdateFrame(frame,dtTime) * 0.3 + frame.ptzAngle;
    serial_ptr->SendPTZAbsoluteAngle(res.x,res.y);
    // test serial
    //serial_ptr->SendSingleOrder(222);
//    cout << "PTZ Angle " << frame.ptzAngle << endl;
//    static float x = 0;
//    if (waitedKey == 'a') x-=0.5f;
//    if (waitedKey == 'd') x += 0.5f;
//    serial_ptr->SendPTZAbsoluteAngle(x,0);
    /*
    static int exposureTime = ConfigurationVariables::GetInt("exposureTime",64);
    if (waitedKey == 'd')
    {
        exposureTime += 4;
        cout << exposureTime << endl;
        SetCameraExposure((false,exposureTime));
    }
    else if (waitedKey == 'a')
    {
        exposureTime -= 4;
        cout << exposureTime << endl;
        SetCameraExposure((false,exposureTime));
    }*/
}

void DSFDetectDebug(ImageData &frame)
{

    static int flag = 0;
    static int angle_yaw = 0;
    static int angle_p = 0;
/*
    if(flag == 1)
    {
        dsf_reco_ptr->GetDsfPosition(frame);
        angle_yaw = frame.ptzAngle.x;
        angle_p = frame.ptzAngle.y;
    }
    if(flag == 2)
        dsf_reco_ptr->UpdateFrame(frame);
*/
//cout<<frame.shootSpeed<<endl;
    dsf_hiter_ptr->UpdateTest(frame,dtTime, flag);
//    flag=2;
    if(waitedKey == 'r') 
        if(flag != 1) flag = 1;
        else flag = 2;

/*
    if(waitedKey == 'a') 
    {
        angle_yaw += 1; 
        serial_ptr->SendPTZAbsoluteAngle(angle_yaw, angle_p);
    }
    if(waitedKey == 'b') 
    {
        angle_yaw -= 1; 
        serial_ptr->SendPTZAbsoluteAngle(angle_yaw, angle_p);
    }
    if(waitedKey == 'c') 
    {
        angle_p += 1; 
        serial_ptr->SendPTZAbsoluteAngle(angle_yaw, angle_p);
    }
    if(waitedKey == 'd') 
    {
       angle_p -= 1; 
       serial_ptr->SendPTZAbsoluteAngle(angle_yaw, angle_p);
    } 
*/
}

void AutoLeadDebug(ImageData &frame)
{

}

void AutoBlockDebug(ImageData &frame)
{
    static bool started = false;
    if (waitedKey == 's' && !started)
    {
        cout << "started " << endl;
        autoBlock_ptr->EnableModule();
        started = true;
    }
    else if (started && waitedKey == 'e')
    {
        autoBlock_ptr->DisableModule();
        started = false;
    }

    if (started)
    {
        autoBlock_ptr->Update(frame,dtTime);
    }
}
