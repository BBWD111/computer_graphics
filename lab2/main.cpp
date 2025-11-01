#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

// ----------------- Utils -----------------
static Mat loadOrBlank(const string& path) {
    if (!path.empty()) {
        Mat img = imread(path, IMREAD_COLOR);
        if (!img.empty()) return img;
        cerr << "Cannot open image: " << path << endl;
    }
    return Mat(512, 512, CV_8UC3, Scalar(16, 16, 16));
}

static Mat toGray(const Mat& bgr) {
    Mat g; cvtColor(bgr, g, COLOR_BGR2GRAY); return g;
}
static Mat grayToBGR(const Mat& g) {
    Mat bgr; cvtColor(g, bgr, COLOR_GRAY2BGR); return bgr;
}

// ----------------- Histogram drawing -----------------
static Mat drawHistGray(const Mat& gray, int hist_w=512, int hist_h=300) {
    CV_Assert(gray.type()==CV_8UC1);
    vector<int> hist(256,0);
    for (int y=0;y<gray.rows;++y) {
        const uchar* p = gray.ptr<uchar>(y);
        for (int x=0;x<gray.cols;++x) hist[p[x]]++;
    }
    int maxv = *max_element(hist.begin(), hist.end());
    Mat img(hist_h, hist_w, CV_8UC3, Scalar(30,30,30));
    // оси
    rectangle(img, Rect(0,0,hist_w-1,hist_h-1), Scalar(90,90,90));
    // рисуем столбцы
    for (int i=0;i<256;++i) {
        int h = (int)round((hist[i]/(double)maxv)*(hist_h-20));
        int x0 = (int)round(i*(hist_w/256.0));
        int x1 = (int)round((i+1)*(hist_w/256.0));
        rectangle(img, Rect(Point(x0, hist_h-10-h), Point(x1, hist_h-10)), Scalar(200,200,200), FILLED);
    }
    return img;
}

static Mat drawHistBGR(const Mat& src, int hist_w=512, int hist_h=300) {
    CV_Assert(src.type()==CV_8UC3);
    vector<int> hb(256,0), hg(256,0), hr(256,0);
    for (int y=0;y<src.rows;++y) {
        const Vec3b* p = src.ptr<Vec3b>(y);
        for (int x=0;x<src.cols;++x) {
            hb[p[x][0]]++; hg[p[x][1]]++; hr[p[x][2]]++;
        }
    }
    int maxb = *max_element(hb.begin(), hb.end());
    int maxg = *max_element(hg.begin(), hg.end());
    int maxr = *max_element(hr.begin(), hr.end());
    int maxv = max({maxb,maxg,maxr});
    Mat img(hist_h, hist_w, CV_8UC3, Scalar(30,30,30));
    rectangle(img, Rect(0,0,hist_w-1,hist_h-1), Scalar(90,90,90));

    auto drawLine = [&](const vector<int>& h, Scalar col){
        Point prev(0, hist_h-10);
        for (int i=0;i<256;++i){
            int hpx = (int)round((h[i]/(double)maxv)*(hist_h-20));
            int x = (int)round(i*(hist_w/256.0));
            Point cur(x, hist_h-10 - hpx);
            if (i>0) line(img, prev, cur, col, 2, LINE_AA);
            prev = cur;
        }
    };
    drawLine(hb, Scalar(200,100,100)); // Blue channel: красноватая линия, чтобы было видно на синем — BGR путаницу избегаем
    drawLine(hg, Scalar(100,200,100));
    drawLine(hr, Scalar(100,100,200));
    return img;
}

// ----------------- Linear contrast (percentile stretch) -----------------
static void calcBoundsGray(const Mat& gray, double lowPct, double highPct, int& lo, int& hi) {
    CV_Assert(gray.type()==CV_8UC1);
    vector<int> hist(256,0);
    const int N = gray.rows*gray.cols;
    for (int y=0;y<gray.rows;++y){
        const uchar* p=gray.ptr<uchar>(y);
        for (int x=0;x<gray.cols;++x) hist[p[x]]++;
    }
    vector<int> cdf(256,0);
    cdf[0]=hist[0];
    for (int i=1;i<256;++i) cdf[i]=cdf[i-1]+hist[i];
    int lowCount  = (int)round(lowPct  * N);
    int highCount = (int)round(highPct * N);
    lo = 0; while (lo<255 && cdf[lo] < lowCount)  ++lo;
    hi = 255; while (hi>0 && cdf[hi] > highCount) --hi;
    if (lo>=hi){ lo=0; hi=255; }
}

static Mat stretchGray(const Mat& gray, double lowPercent, double highPercent) {
    int lo,hi;
    calcBoundsGray(gray, lowPercent, 1.0 - highPercent, lo, hi);
    Mat out(gray.size(), CV_8UC1);
    const double scale = (hi>lo) ? 255.0/(hi-lo) : 1.0;
    for (int y=0;y<gray.rows;++y){
        const uchar* s=gray.ptr<uchar>(y);
        uchar* d=out.ptr<uchar>(y);
        for (int x=0;x<gray.cols;++x){
            int v = (int)round((s[x]-lo)*scale);
            d[x] = (uchar)std::clamp(v,0,255);
        }
    }
    return out;
}

static Mat stretchPerChannel(const Mat& src, double lowPercent, double highPercent) {
    vector<Mat> ch; split(src, ch);
    for (int k=0;k<3;++k) ch[k] = stretchGray(ch[k], lowPercent, highPercent);
    Mat out; merge(ch, out);
    return out;
}

// ----------------- Histogram equalization -----------------
static Mat equalizeGray(const Mat& src) {
    Mat g = (src.channels()==1)?src:toGray(src);
    Mat e; equalizeHist(g, e);
    return (src.channels()==1)?e:grayToBGR(e);
}
static Mat equalizeRGB(const Mat& src) {
    CV_Assert(src.type()==CV_8UC3);
    vector<Mat> ch; split(src, ch);
    for (auto& c : ch) equalizeHist(c, c);
    Mat out; merge(ch, out); return out;
}
static Mat equalizeHSV_V(const Mat& src) {
    CV_Assert(src.type()==CV_8UC3);
    Mat hsv; cvtColor(src, hsv, COLOR_BGR2HSV);
    vector<Mat> ch; split(hsv, ch);
    equalizeHist(ch[2], ch[2]); // V
    Mat out; merge(ch, hsv); cvtColor(hsv, out, COLOR_HSV2BGR);
    return out;
}
static Mat equalizeHLS_L(const Mat& src) {
    CV_Assert(src.type()==CV_8UC3);
    Mat hls; cvtColor(src, hls, COLOR_BGR2HLS);
    vector<Mat> ch; split(hls, ch);
    equalizeHist(ch[1], ch[1]); // L
    Mat out; merge(ch, hls); cvtColor(hls, out, COLOR_HLS2BGR);
    return out;
}

// ----------------- Pointwise ops (per-element) -----------------
static Mat invert(const Mat& src){
    Mat out; bitwise_not(src, out); return out;
}
static Mat gammaCorr(const Mat& src, double gamma){
    CV_Assert(gamma>0);
    // LUT по яркости для каждого канала (8-бит)
    vector<uchar> lut(256);
    for (int i=0;i<256;++i){
        double v = pow(i/255.0, 1.0/gamma)*255.0;
        lut[i] = (uchar)std::clamp((int)round(v),0,255);
    }
    Mat out = src.clone();
    if (src.channels()==1){
        LUT(src, Mat(1,256,CV_8U,lut.data()), out);
    } else {
        vector<Mat> ch; split(src, ch);
        for (auto& c: ch) LUT(c, Mat(1,256,CV_8U,lut.data()), c);
        merge(ch, out);
    }
    return out;
}
static Mat addConst(const Mat& src, int beta){
    Mat out; src.convertTo(out, -1, 1.0, beta); return out;
}
static Mat mulConst(const Mat& src, double alpha){
    Mat out; src.convertTo(out, -1, alpha, 0.0); return out;
}

// ----------------- UI/State -----------------
struct State {
    int mode = 0;     // см. список в UI
    int p1 = 1;       // общий параметр 1
    int p2 = 99;      // общий параметр 2
    Mat src, cur, hist;
} st;

static void recompute(){
    // интерпретация параметров
    int m = st.mode;
    // для линейного контраста используем отсечки low / high в процентах
    double lowCut = std::clamp(st.p1, 0, 20) / 100.0;     // 0..0.20
    double highCut = (100 - std::clamp(st.p2, 80, 100)) / 100.0; // 0..0.20

    switch(m){
        case 0: st.cur = st.src.clone(); break;
        case 1: { // Linear Contrast (Gray)
            Mat g = (st.src.channels()==1)?st.src:toGray(st.src);
            st.cur = grayToBGR(stretchGray(g, lowCut, highCut));
        } break;
        case 2: { // Linear Contrast (Per-Channel)
            Mat tmp = (st.src.channels()==1)?grayToBGR(st.src):st.src;
            st.cur = stretchPerChannel(tmp, lowCut, highCut);
        } break;
        case 3: st.cur = equalizeGray(st.src); break;
        case 4: {
            Mat tmp = (st.src.channels()==1)?grayToBGR(st.src):st.src;
            st.cur = equalizeRGB(tmp);
        } break;
        case 5: {
            Mat tmp = (st.src.channels()==1)?grayToBGR(st.src):st.src;
            st.cur = equalizeHSV_V(tmp);
        } break;
        case 6: {
            Mat tmp = (st.src.channels()==1)?grayToBGR(st.src):st.src;
            st.cur = equalizeHLS_L(tmp);
        } break;
        case 7: st.cur = invert(st.src); break;
        case 8: { // Gamma
            double gamma = std::max(0.01, st.p1/100.0); // p1=100 -> gamma=1.0
            st.cur = gammaCorr(st.src, gamma);
        } break;
        case 9: { // Add/Sub
            int beta = st.p1 - 128; // сдвиг: [0..255] -> [-128..+127]
            st.cur = addConst(st.src, beta);
        } break;
        case 10:{ // Multiply
            double alpha = std::max(0.0, st.p1/100.0); // 100 -> 1.0
            st.cur = mulConst(st.src, alpha);
        } break;
        default: st.cur = st.src.clone(); break;
    }

    // гистограмма для текущего результата
    Mat show = st.cur;
    if (show.channels()==3){
        st.hist = drawHistBGR(show);
    } else {
        st.hist = drawHistGray(show);
    }

    imshow("Image", st.cur);
    imshow("Histogram", st.hist);
}

static void on_trackbar(int, void*) { recompute(); }

int main(int argc, char** argv){
    string path = (argc>1)? argv[1] : "";
    st.src = loadOrBlank(path);
    if (st.src.channels()!=1 && st.src.channels()!=3){
        cerr << "Only 8-bit grayscale or BGR images are supported.\n";
        return 1;
    }

    namedWindow("Image", WINDOW_AUTOSIZE);
    namedWindow("Histogram", WINDOW_AUTOSIZE);
    namedWindow("Controls", WINDOW_AUTOSIZE);

    // трекбары
    createTrackbar("Mode (0..10)", "Controls", &st.mode, 10, on_trackbar);
    // P1/P2 по умолчанию для линейного контрастирования: 1%/99%
    st.p1 = 1;  st.p2 = 99;
    createTrackbar("P1", "Controls", &st.p1, 255, on_trackbar);
    createTrackbar("P2", "Controls", &st.p2, 255, on_trackbar);

    // первые подсказки по значениям:
    // Mode=1/2: P1=low% (0..20), P2=high% (80..100)
    // Mode=8 (Gamma): P1=gamma*100  (например 220 => γ=2.20)
    // Mode=9 (Add/Sub): P1 in [0..255] -> beta=P1-128
    // Mode=10 (Multiply): P1=alpha*100 (120 => *1.20)

    recompute();

    cout << "Controls:\n"
         << "  - Use trackbars in 'Controls' window.\n"
         << "  - Press 's' to save result.png\n"
         << "  - Press 'q' or ESC to quit\n";

    for(;;){
        int k = waitKey(15);
        if (k==27 || k=='q') break;
        if (k=='s'){
            imwrite("result.png", st.cur);
            cout << "Saved: result.png\n";
        }
    }
    destroyAllWindows();
    return 0;
}
