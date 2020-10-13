#include "BFilter.h"
#include "GFilter.h"
#include "guidedfilter.h"

//#define FILTERR 10
//#define EPS 0.01
// #define FILTERR 20
// #define EPS 0.1

int main(int argc, char *argv[]) {
    //std::cout << "Hello, World!" << std::endl;

    clock_t start, stop;

    //Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
    Mat img_g = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img_p = imread(argv[2], IMREAD_GRAYSCALE);
    int FILTERR = atoi(argv[3]);
    float EPS = atof(argv[4]);
    try
    {
        if(!img_p.data or !img_g.data)
            throw runtime_error("Read Image failed...");
    }
    catch(runtime_error err)
    {
        cerr << err.what() << endl;
        return -1;
    }

    int row = img_p.rows;
    int col = img_p.cols;

    cout << "Image Info: row = " << row << ", col = " << col << endl;

    imshow("Guidance", img_g);
    imshow("Input", img_p);

    img_g.convertTo(img_g, CV_32F, 1.0);
    img_p.convertTo(img_p, CV_32F, 1.0/1000);

    float *imgGui = (float *)img_g.data;
    float *imgInP = (float *)img_p.data;

    Mat imgOut = Mat::zeros(Size(col, row), CV_32F);
    float *imgOutP = (float *)imgOut.data;

    //BFilter bf(col, row);
    //bf.boxfilterTest(imgOutP, imgInP, col, row, FILTERR);
    GFilter gf(col, row);

    start = clock();
    gf.guidedfilterTest(imgOutP, imgGui, imgInP, col, row, FILTERR, EPS);
    stop = clock();
    cout << "GPU Time : " << (stop - start) * 1.0 / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;

    // Output For test
    /*
    for(int i = 0; i < 10; ++i)
        cout << imgOutP[i] << endl;
    */

    imgOut.convertTo(imgOut, CV_8UC1, 1.0*1000);
    imshow("GPU Output", imgOut);
    // imwrite("gpuout.png", imgOut);

    // guided filter on OpenCV
    Mat OpenCV_ImgOut = Mat::zeros(Size(row, col), CV_32F);
    start = clock();
    OpenCV_ImgOut = guidedFilter(img_g, img_p, FILTERR, EPS, CV_32F);
    stop = clock();
    cout << "OpenCV Time : " << (stop - start) * 1.0 / CLOCKS_PER_SEC * 1000.0 << " ms" << endl;
    OpenCV_ImgOut.convertTo(OpenCV_ImgOut, CV_8UC1, 1.0*1000);
    imshow("OpenCV Output", OpenCV_ImgOut);
    // imwrite("opencvout.png", OpenCV_ImgOut);
    
    waitKey(0);

    return 0;
}
