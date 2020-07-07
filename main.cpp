#include <utility>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/framework/tensor.h"
#include "google/protobuf/wrappers.pb.h"
#include "tensorflow/core/util/command_line_flags.h"

using tensorflow::string;
using tensorflow::Tensor;

using namespace tensorflow;

#include <dirent.h>
const int fixedSize=32;
int total=0;
int good=0;
int bad=0;
/*模型路径*/
const string model_path = "../Model/happyModel.pb";
/*输入输出节点详见ipynb的summary*/
const string input_name = "input_1:0";
const string output_name = "y/Sigmoid:0";
/*计时*/
class timer
{
public:
    using clk_t = std::chrono::high_resolution_clock;
    timer() : t(clk_t::now()) {}
    void reset(){  t = clk_t::now(); }

    double milli_cnt() const
    {
        return std::chrono::duration<double, std::milli>(clk_t::now() - t).count();
    }

private:
    clk_t::time_point t;
}; // class timer

/************************
*mat2Tensor
*params @ Mat &image(输入), Tensor &t(tf的输入)
*func @ 适用于神经网络的输入
************************/
void mat2Tensor(cv::Mat &image, Tensor &t) {
    float *tensor_data_ptr = t.flat<float>().data();
    cv::Mat fake_mat(image.rows, image.cols, CV_32FC(image.channels()), tensor_data_ptr);
    image.convertTo(fake_mat, CV_32FC(image.channels()));
}

//void tensor2Mat(Tensor &t, cv::Mat &image) {
//    int *p = t.flat<int>().data();
//    image = cv::Mat(IMAGE_SIZE, IMAGE_SIZE, CV_32SC1, p);
//    image.convertTo(image, CV_8UC1);
//}

/************************
**getThreshold
**图片取得阈值
**params @ 一目了然
**func @ 获得前百分比高光
************************/

int getThreshold(const cv::Mat& mat,double thre_proportion=0.1){

    uint32_t iter_rows = mat.rows;
    uint32_t iter_cols = mat.cols;
    auto sum_pixel = iter_rows * iter_cols;
    if(mat.isContinuous()){
        iter_cols = sum_pixel;
        iter_rows = 1;
    }
    int histogram[256];
    memset(histogram, 0, sizeof(histogram));//置零
    for (uint32_t i = 0; i < iter_rows; ++i){
        const auto* lhs = mat.ptr<uchar>(i);
        for (uint32_t j = 0; j < iter_cols; ++j)
            ++histogram[*lhs++];
    }

    auto left = thre_proportion * sum_pixel;
    int i = 255;
    while((left -= histogram[i--]) > 0);
    return i>0?i:0;
}

/************************
**loadAndPre
**载入与预处理
**params @ string& address(载入地址) Mat &result(输出结果)
**func @ 载入与预处理
************************/

bool loadAndPre(const std::string& address,cv::Mat &result){
    //注意已经灰度化了
    cv::Mat img =imread(address,cv::IMREAD_GRAYSCALE);
    //cout<<img.cols<<" "<<img.rows<<endl;
    if(img.cols==0)
        return false;
    //调整大小 同比缩放至fixedsize*fixedsize以内
    if(img.cols<img.rows)
        resize(img,img,{int(img.cols*1.0/img.rows*fixedSize),fixedSize});
    else
        resize(img,img,{fixedSize,int(img.rows*1.0/img.cols*fixedSize)});

    //剪去边上多余部分
    int cutRatio1=0.15*img.cols;
    int cutRatio2=0.05*img.rows;
    cv::Mat blank=cv::Mat(cv::Size(fixedSize,fixedSize), img.type(), cv::Scalar(0));//新建空白
    cv::Mat mask=img(cv::Rect(cutRatio1,cutRatio2,img.cols-2*cutRatio1,img.rows-2*cutRatio2));//建立腌摸
    cv::Mat imageROI=blank(cv::Rect(cutRatio1,cutRatio2,img.cols-2*cutRatio1,img.rows-2*cutRatio2));//建立需要覆盖区域的ROI
    mask.copyTo(imageROI, mask);

    //imshow("mask",mask);//小图
    //imshow("blank",blank);//大图

    int thre=getThreshold(blank);//均值获取阈值
    result=blank.clone();
    //补高光，而不直接粗暴二值化
    for (int i = 0; i<result.rows; i++){
        for (int j = 0; j<result.cols; j++){
            if((int)result.at<u_char>(i, j)>thre){
                result.at<u_char>(i, j)=200;
            }
        }
    }
    //imshow("result",result);
    //cv::waitKey();
    return true;
}

/************************
**init_my_tf
**载入与预处理
**params @ Session* session(为网络声明session)
**func @ 载入与预处理
**return @ Tensor input 网络输入口 注意TensorShape(与输入的网络结构一致)
************************/
inline Tensor init_my_tf(Session* session){
    /*--------------------------------从pb文件中读取模型--------------------------------*/

    GraphDef graph_def;
    //读取Graph, 如果是文本形式的pb,使用ReadTextProto
    Status status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    } else {
        std::cout << "Load graph protobuf successfully" << std::endl;
    }
    /*--------------------------------将模型设置到创建的Session里--------------------------------*/
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    } else {
        std::cout << "Add graph to session successfully" << std::endl;
    }

    Tensor input(DT_FLOAT, TensorShape({ 1, fixedSize, fixedSize, 1 }));
    return input;

}


int main(){
    std::cout<<"hello world"<<std::endl;
//1、载入权重信息
    Session* session;
    /*--------------------------------创建session------------------------------*/
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    } else {
        std::cout << "Session created successfully" << std::endl;
    }
/*初始化*/
    Tensor input=init_my_tf(session);

/*测试数据准备*/
    bool saveResult= false;//true;

    std::string where="../data/training/2/";
    DIR *dir_ptr = opendir(where.c_str());
    dirent *dptr;
    timer t;
    std::string confirDir="/home/truth/ClionProjects/RM/分类器专用/NN/CNN/myCNN/result/";

    while ((dptr = readdir(dir_ptr)) != nullptr) {
        if (dptr->d_name[0] == '.')
            continue;
        //std::cout<<dptr->d_name<<": ";
        cv::Mat image;
//2、载入待测图片信息
        if(loadAndPre(where+dptr->d_name,image)){
            //形式的转换
            mat2Tensor(image, input);
/*保留最终输出*/
            std::vector<tensorflow::Tensor> outputs;
// 3、计算最后结果
            TF_CHECK_OK(session->Run({std::pair<string, Tensor>(input_name, input)}, {output_name}, {}, &outputs));
            //获取输出
            auto output_c = outputs[0].scalar<float>();
            float result = output_c();
/*判断正负样本*/
            std::cout<<dptr->d_name<<"  "<<"result :"<<result<<std::endl;
            if(0.5<result){
                good++;
                if(saveResult){
                    cv::Mat temp=cv::imread(where+dptr->d_name);
                    imwrite(confirDir+"pos/"+dptr->d_name,temp);
                }
            }
            else{
                bad++;
                if(saveResult) {
                    cv::Mat temp = cv::imread(where + dptr->d_name);
                    imwrite(confirDir + "neg/" + dptr->d_name, temp);
                }
            }
            total++;
        }
        else
            continue;
    }

    std::cout << "Average bench time: " << t.milli_cnt() / total << " ms\n";

    std::cout<<"total ："<<total<<std::endl;
    std::cout<<"good ："<<good<<std::endl;
    std::cout<<"bad ："<<bad<<std::endl;
    std::cout<<"正确率 ："<<std::setprecision(3)<<static_cast<double>(good*1.0/total)*100<<"%"<<std::endl;
    std::cout<<"正确率 ："<<std::setprecision(3)<<static_cast<double>(bad*1.0/total)*100<<"%"<<std::endl;

    session->Close();
    return 0;

}

