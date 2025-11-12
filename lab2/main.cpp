// lab2_full.cpp — Pointwise ops + Linear Contrast + Equalization (RGB & HSV/HLS) + nice histograms
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

namespace fs = std::filesystem;
using namespace cv;
using std::string;
using std::vector;

enum class Mode {
    ORIGINAL = 0,
    LC_HSV_V,
    LC_RGB,
    EQ_GRAY,
    EQ_RGB,
    EQ_HSV_V,
    EQ_HLS_L,
    NEGATIVE,
    ADD_BIAS,
    MULTIPLY_ALPHA,
    GAMMA,
    LOGARITHM
};

struct UIState {
    int mode = static_cast<int>(Mode::ORIGINAL);

    int use_clip  = 1;
    int clip_x10  = 10;  // 0..200 → 0..20.0 (%)

    // поэлементные
    int bias_shifted = 128;    // 0..256 -> bias = raw-128
    int alpha_x100   = 100;    // 10..300 -> 0.10..3.00
    int gamma_x100   = 100;    // 10..300 -> 0.10..3.00
    int log_c_x100   = 100;    // 1..500  -> 0.01..5.00

    // прочее
    int show_hist    = 1;
};

static bool endsWithAny(const string& s, std::initializer_list<string> exts){
    string low=s; std::transform(low.begin(),low.end(),low.begin(),::tolower);
    for (auto& e: exts) if (low.size()>=e.size() && low.rfind(e)==low.size()-e.size()) return true;
    return false;
}
static vector<string> collectImages(const string& arg){
    vector<string> files; if (arg.empty()) return files;
    fs::path p(arg); if (!fs::exists(p)) return files;
    if (fs::is_directory(p)){
        for (auto& de: fs::directory_iterator(p)){
            if (!de.is_regular_file()) continue;
            string f=de.path().string();
            if (endsWithAny(f,{".jpg",".jpeg",".png",".bmp",".tif",".tiff"})) files.push_back(f);
        }
        std::sort(files.begin(), files.end());
    } else if (endsWithAny(arg,{".jpg",".jpeg",".png",".bmp",".tif",".tiff"})) {
        files.push_back(arg);
    }
    return files;
}
static void ensureDir(const fs::path& p){ if(!fs::exists(p)) fs::create_directories(p); }

static void drawAxes(Mat& img, int left, int right, int bottom, int top){
    line(img, Point(left, bottom), Point(right, bottom), Scalar(150,150,150), 1, LINE_AA);
    line(img, Point(left, bottom), Point(left,  top),    Scalar(150,150,150), 1, LINE_AA);
    for (int x=0; x<=256; x+=64){
        int px = left + (int)std::round(x*(right-left)/256.0);
        line(img, Point(px, bottom), Point(px, top), Scalar(60,60,60), 1, LINE_AA);
    }
    for (int i=1;i<=4;i++){
        int py = bottom - (int)std::round(i*(bottom-top)/4.0);
        line(img, Point(left, py), Point(right, py), Scalar(60,60,60), 1, LINE_AA);
    }
}

static Mat drawHistGrayNice(const Mat& gray, int W=640, int H=360){
    CV_Assert(gray.type()==CV_8U);
    vector<int> hist(256,0);
    for(int y=0;y<gray.rows;++y){
        const uchar* r=gray.ptr<uchar>(y);
        for(int x=0;x<gray.cols;++x) hist[r[x]]++;
    }
    int maxv=*std::max_element(hist.begin(),hist.end()); if(maxv<=0) maxv=1;

    Mat img(H,W,CV_8UC3,Scalar(30,30,30));
    int left=50, right=W-20, bottom=H-40, top=30;
    drawAxes(img,left,right,bottom,top);

    // сглаживание
    vector<double> curve(256);
    for (int i=0;i<256;++i) curve[i] = hist[i]/(double)maxv;

    // кривая
    Point prev(left, bottom);
    for (int i=0;i<256;++i){
        double y = curve[i];
        int px = left + (int)std::round(i*(right-left)/256.0);
        int py = bottom - (int)std::round(y*(bottom-top));
        Point cur(px,py);
        if (i>0) line(img, prev, cur, Scalar(220,220,220), 2, LINE_AA);
        prev = cur;
    }
    putText(img, "Gray histogram", Point(left, top-8), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(220,220,220), 2, LINE_AA);
    return img;
}

static Mat drawHistBGRNice(const Mat& bgr, int W=640, int H=360){
    CV_Assert(bgr.type()==CV_8UC3);
    vector<int> hb(256,0), hg(256,0), hr(256,0);
    for(int y=0;y<bgr.rows;++y){
        const Vec3b* r=bgr.ptr<Vec3b>(y);
        for(int x=0;x<bgr.cols;++x){
            hb[r[x][0]]++; hg[r[x][1]]++; hr[r[x][2]]++;
        }
    }
    int maxv=std::max({*std::max_element(hb.begin(),hb.end()),
                       *std::max_element(hg.begin(),hg.end()),
                       *std::max_element(hr.begin(),hr.end())});
    if(maxv<=0) maxv=1;

    Mat img(H,W,CV_8UC3,Scalar(30,30,30));
    int left=50, right=W-20, bottom=H-40, top=30;
    drawAxes(img,left,right,bottom,top);

    auto plot=[&](const vector<int>& h, Scalar col){
        Point prev(left,bottom);
        for(int i=0;i<256;++i){
            double y=h[i]/(double)maxv;
            int px = left + (int)std::round(i*(right-left)/256.0);
            int py = bottom - (int)std::round(y*(bottom-top));
            Point cur(px,py);
            if (i>0) line(img, prev, cur, col, 2, LINE_AA);
            prev=cur;
        }
    };
    plot(hb, Scalar(255,140,140)); // B
    plot(hg, Scalar(140,255,140)); // G
    plot(hr, Scalar(140,140,255)); // R
    putText(img, "BGR histograms", Point(left, top-8), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(230,230,230), 2, LINE_AA);
    return img;
}

static void stretchGray_inplace(Mat& gray, double clipPercent){
    CV_Assert(gray.type()==CV_8U);
    const int N=gray.rows*gray.cols;
    vector<int> hist(256,0);
    for (int y=0;y<gray.rows;++y){
        const uchar* r=gray.ptr<uchar>(y);
        for (int x=0;x<gray.cols;++x) hist[r[x]]++;
    }
    vector<int> cdf(256,0); cdf[0]=hist[0];
    for(int i=1;i<256;++i) cdf[i]=cdf[i-1]+hist[i];

    int cut = (int)std::round(N*(clipPercent/100.0));
    int fmin=0; while(fmin<255 && cdf[fmin] <= cut) ++fmin;
    int fmax=255; while(fmax>0 && (N - cdf[fmax-1]) <= cut) --fmax;
    if (fmax<=fmin) return;

    vector<uchar> lut(256);
    int denom = std::max(1, fmax - fmin);
    for (int i=0;i<256;++i){
        int v = (int)std::round((i - fmin)*255.0/denom);
        lut[i] = (uchar)saturate_cast<uchar>(v);
    }
    LUT(gray, Mat(1,256,CV_8U,lut.data()), gray);
}

static Mat linearContrastHSV_V(const Mat& bgr, double clipPercent){
    Mat hsv; cvtColor(bgr,hsv,COLOR_BGR2HSV);
    vector<Mat> ch; split(hsv,ch);
    stretchGray_inplace(ch[2], clipPercent);
    merge(ch,hsv);
    Mat out; cvtColor(hsv,out,COLOR_HSV2BGR); return out;
}
static Mat linearContrastRGB(const Mat& bgr, double clipPercent){
    vector<Mat> ch; split(bgr,ch);
    for (auto& c: ch) stretchGray_inplace(c, clipPercent);
    Mat out; merge(ch,out); return out;
}

static Mat equalizeGrayBGR(const Mat& src){
    Mat g; cvtColor(src,g,COLOR_BGR2GRAY);
    Mat e; equalizeHist(g,e);
    Mat out; cvtColor(e,out,COLOR_GRAY2BGR); return out;
}
static Mat equalizeRGB(const Mat& src){
    vector<Mat> ch; split(src,ch); for (auto& c: ch) equalizeHist(c,c);
    Mat out; merge(ch,out); return out;
}
static Mat equalizeHSV_V(const Mat& src){
    Mat hsv; cvtColor(src,hsv,COLOR_BGR2HSV);
    vector<Mat> ch; split(hsv,ch); equalizeHist(ch[2],ch[2]); merge(ch,hsv);
    Mat out; cvtColor(hsv,out,COLOR_HSV2BGR); return out;
}
static Mat equalizeHLS_L(const Mat& src){
    Mat hls; cvtColor(src,hls,COLOR_BGR2HLS);
    vector<Mat> ch; split(hls,ch); equalizeHist(ch[1],ch[1]); merge(ch,hls);
    Mat out; cvtColor(hls,out,COLOR_HLS2BGR); return out;
}

static Mat opNegative(const Mat& src){ Mat out; bitwise_not(src,out); return out; }
static Mat opAddBias(const Mat& src, int bias){ Mat out; src.convertTo(out, CV_8U, 1.0, bias); return out; }
static Mat opMulAlpha(const Mat& src, double a){ Mat out; src.convertTo(out, CV_8U, a, 0.0); return out; }
static Mat opGamma(const Mat& src, double gamma){
    gamma = std::max(0.001, gamma);
    vector<uchar> lut(256);
    for (int i=0;i<256;++i){
        double v = std::pow(i/255.0, 1.0/gamma)*255.0;
        lut[i] = (uchar)saturate_cast<uchar>(std::lround(v));
    }
    Mat out; LUT(src, Mat(1,256,CV_8U,lut.data()), out); return out;
}
static Mat opLog(const Mat& src, double c){
    c = std::max(0.01, c);
    Mat f32; src.convertTo(f32, CV_32F, 1.0/255.0);
    Mat g32; log(f32 + 1.0, g32); g32 *= c;
    // общая нормировка, чтобы не «ломать» цвет
    double mn, mx; minMaxLoc(g32, &mn, &mx);
    Mat norm = (g32 - mn) / (mx - mn + 1e-9);
    Mat out8; norm.convertTo(out8, CV_8U, 255.0); return out8;
}

static Mat applyMode(const Mat& inputBGR, const UIState& st){
    double clipPercent = (st.use_clip? std::clamp(st.clip_x10,0,200)/10.0 : 0.0);

    switch ((Mode)st.mode){
        case Mode::ORIGINAL:   return inputBGR;
        case Mode::LC_HSV_V:   return linearContrastHSV_V(inputBGR, clipPercent);
        case Mode::LC_RGB:     return linearContrastRGB(inputBGR, clipPercent);
        case Mode::EQ_GRAY:    return equalizeGrayBGR(inputBGR);
        case Mode::EQ_RGB:     return equalizeRGB(inputBGR);
        case Mode::EQ_HSV_V:   return equalizeHSV_V(inputBGR);
        case Mode::EQ_HLS_L:   return equalizeHLS_L(inputBGR);
        case Mode::NEGATIVE:   return opNegative(inputBGR);
        case Mode::ADD_BIAS:   return opAddBias(inputBGR, st.bias_shifted - 128);
        case Mode::MULTIPLY_ALPHA:
            return opMulAlpha(inputBGR, std::clamp(st.alpha_x100,10,300)/100.0);
        case Mode::GAMMA:
            return opGamma(inputBGR,  std::clamp(st.gamma_x100,10,300)/100.0);
        case Mode::LOGARITHM:
            return opLog(inputBGR,    std::clamp(st.log_c_x100,1,500)/100.0);
    }
    return inputBGR;
}

static string modeName(Mode m){
    switch(m){
        case Mode::ORIGINAL:  return "Original";
        case Mode::LC_HSV_V:  return "Linear Contrast (HSV-V)";
        case Mode::LC_RGB:    return "Linear Contrast (RGB per-channel)";
        case Mode::EQ_GRAY:   return "Equalize (Gray)";
        case Mode::EQ_RGB:    return "Equalize (RGB)";
        case Mode::EQ_HSV_V:  return "Equalize (HSV-V)";
        case Mode::EQ_HLS_L:  return "Equalize (HLS-L)";
        case Mode::NEGATIVE:  return "Negative";
        case Mode::ADD_BIAS:  return "Add/Sub (bias)";
        case Mode::MULTIPLY_ALPHA: return "Multiply alpha";
        case Mode::GAMMA:     return "Gamma";
        case Mode::LOGARITHM: return "Log";
    } return "?";
}

int main(int argc, char** argv){
    string arg = (argc>1)? argv[1] : "";
    auto files = collectImages(arg.empty()? "samples" : arg);
    if (files.empty()){
        std::cerr << "Нет изображений. Передай путь к файлу/папке (jpg/png/bmp).\n";
        return 1;
    }

    UIState st;
    const string WIN = "Lab2: Point Ops + Linear Contrast + Equalization";
    namedWindow(WIN, WINDOW_NORMAL); resizeWindow(WIN, 1220, 760);

    createTrackbar("Mode (0..11)", WIN, &st.mode, 11);

    // линейный контраст
    createTrackbar("LC: use clip% (0/1)", WIN, &st.use_clip, 1);
    createTrackbar("LC: clip% x10 (0..200)", WIN, &st.clip_x10, 200);

    // поэлементные
    createTrackbar("Bias [-128..128]", WIN, &st.bias_shifted, 256);
    setTrackbarPos("Bias [-128..128]", WIN, st.bias_shifted);
    createTrackbar("Alpha x100 [10..300]", WIN, &st.alpha_x100, 300);
    createTrackbar("Gamma x100 [10..300]", WIN, &st.gamma_x100, 300);
    createTrackbar("Log C x100 [1..500]", WIN, &st.log_c_x100, 500);

    createTrackbar("Show histogram (0/1)", WIN, &st.show_hist, 1);

    int idx=0; fs::path outDir="output"; ensureDir(outDir);

    for(;;){
        Mat src = imread(files[idx], IMREAD_COLOR);
        if (src.empty()){ std::cerr<<"Не открыть: "<<files[idx]<<"\n"; break; }

        Mat res = applyMode(src, st);

        Mat vis; hconcat(src, res, vis);
        string status = "Image: " + fs::path(files[idx]).filename().string()
                      + " | Mode: " + modeName((Mode)st.mode)
                      + " | Keys: TAB-next, H-hist, S-save, \u2190/\u2192 switch, ESC-exit";
        imshow(WIN, vis); displayStatusBar(WIN, status, 0);

        if (st.show_hist){
            Mat hist = drawHistBGRNice(res);
            imshow("Histogram (result)", hist);
        } else {
            destroyWindow("Histogram (result)");
        }

        int key = waitKey(30);
        if (key==27 || key=='q' || key=='Q') break;
        if (key=='\t'){ st.mode = (st.mode+1)%12; }
        if (key=='h' || key=='H'){ st.show_hist = !st.show_hist; }
        if (key=='s' || key=='S'){
            fs::path p = outDir/(fs::path(files[idx]).stem().string()+"_"+modeName((Mode)st.mode)+".png");
            imwrite(p.string(), res);
            std::cout<<"Saved: "<<p<<"\n";
        }
        if (key==81 || key=='a'){ idx = (idx - 1 + (int)files.size()) % (int)files.size(); }
        if (key==83 || key=='d'){ idx = (idx + 1) % (int)files.size(); }
    }
    destroyAllWindows();
    return 0;
}
