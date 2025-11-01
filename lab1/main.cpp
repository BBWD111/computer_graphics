#include <opencv2/opencv.hpp>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// ======= CMYK <-> RGB =======
static inline Vec4f rgb2cmyk(const Vec3b& bgr){
    float r=bgr[2]/255.f, g=bgr[1]/255.f, b=bgr[0]/255.f;
    float k = 1.f - max({r,g,b});
    if (k >= 0.999999f) return {0,0,0,1};
    float d = 1.f - k;
    float c = (1.f - r - k)/d;
    float m = (1.f - g - k)/d;
    float y = (1.f - b - k)/d;
    return {clamp(c,0.f,1.f), clamp(m,0.f,1.f), clamp(y,0.f,1.f), clamp(k,0.f,1.f)};
}

static inline Vec3b cmyk2rgb(const Vec4f& cmyk){
    float C = std::clamp(cmyk[0], 0.f, 1.f);
    float M = std::clamp(cmyk[1], 0.f, 1.f);
    float Y = std::clamp(cmyk[2], 0.f, 1.f);
    float K = std::clamp(cmyk[3], 0.f, 1.f);

    float Rf = 255.f * (1.f - C) * (1.f - K);
    float Gf = 255.f * (1.f - M) * (1.f - K);
    float Bf = 255.f * (1.f - Y) * (1.f - K);

    auto to_u8 = [](float v)->uchar{
        int i = (int)std::lround(v);
        if(i<0) i=0; if(i>255) i=255;
        return (uchar)i;
    };
    return Vec3b(to_u8(Bf), to_u8(Gf), to_u8(Rf)); // BGR
}

// ======= UI / STATE =======
static const string WIN="Color Tool";
static Mat   canvas(760, 1200, CV_8UC3);
static Rect  paletteRect(60, 150, 860, 280);
static Point paletteMarker{paletteRect.x, paletteRect.y};
static bool  draggingPalette=false;
static int   dragCounter = 0;  // ограничение частоты обновления при драге

// значения
static double Rv=92, Gv=204, Bv=52;
static double Hv=52, Lv=50, Sv=60;
static double Cv=54.9, Mv=0.0, Yv=74.51, Kv=20.0;

static inline Mat bgr1x1(){ return Mat(1,1,CV_8UC3, Scalar((int)round(Bv),(int)round(Gv),(int)round(Rv))); }

// ==== КЭШ палитры HS по текущему L ====
static Mat hsCache;
static int hsCacheLcv = -1;

static void drawPanelBg(Mat& img, const Rect& r){
    rectangle(img, r, Scalar(50,50,50), FILLED);
    rectangle(img, r, Scalar(100,100,100), 1, LINE_AA);
}
static void putTextShadow(Mat& img, const string& s, Point org, double fs, Scalar fg){
    putText(img, s, {org.x+1,org.y+2}, FONT_HERSHEY_SIMPLEX, fs, Scalar(0,0,0), 3, LINE_AA);
    putText(img, s, org, FONT_HERSHEY_SIMPLEX, fs, fg, 2, LINE_AA);
}
static void putSmall(Mat& img, const string& s, Point org, Scalar fg=Scalar(245,245,245)){
    putText(img, s, org, FONT_HERSHEY_SIMPLEX, 0.65, fg, 2, LINE_AA);
}

// ======= Поля + слайдеры =======
struct Field{
    string  label;
    Rect    box;
    Rect    slider;
    double  minv, maxv;
    double* value;
    int     group;      // 0=RGB,1=CMYK,2=HLS

    Field(string lbl, Rect b, double lo, double hi, double* ptr, int grp)
        : label(std::move(lbl)), box(b), slider(), minv(lo), maxv(hi), value(ptr), group(grp) {}
};
static vector<Field> fields;

static Rect rgbPanel, cmykPanel, hlsPanel;

enum class EditState{ None, Editing, DragSlider };
static EditState editState = EditState::None;
static int       editIndex = -1;
static string    inputBuf;

// ======= HLS helpers =======
static inline Vec3b HLS_user_to_cv(double Hdeg, double Lpct, double Spct){
    int H = (int)round(fmod(max(0.0, Hdeg), 360.0) / 2.0);
    int L = (int)round(clamp(Lpct, 0.0, 100.0) * 255.0 / 100.0);
    int S = (int)round(clamp(Spct, 0.0, 100.0) * 255.0 / 100.0);
    return Vec3b((uchar)H,(uchar)L,(uchar)S);
}
static inline void HLS_cv_to_user(const Vec3b& cvHLS, double& Hdeg, double& Lpct, double& Spct){
    Hdeg = cvHLS[0] * 2.0;
    Lpct = cvHLS[1] * 100.0 / 255.0;
    Spct = cvHLS[2] * 100.0 / 255.0;
}

// ======= Слайдер =======
static void drawSlider(Mat& img, const Field& f){
    rectangle(img, f.slider, Scalar(30,30,30), FILLED);
    rectangle(img, f.slider, Scalar(120,120,120), 1, LINE_AA);
    double t = (*f.value - f.minv) / (f.maxv - f.minv);
    t = std::clamp(t, 0.0, 1.0);
    int x = f.slider.x + (int)round(t * (f.slider.width-1));
    Rect thumb(max(f.slider.x, x-6), f.slider.y-2, 12, f.slider.height+4);
    rectangle(img, thumb, Scalar(180,180,180), FILLED);
    rectangle(img, thumb, Scalar(40,40,40), 1, LINE_AA);
}

// ======= Синхронизация =======
static void sync_from_rgb(bool redraw=true){
    Mat hls; cvtColor(bgr1x1(), hls, COLOR_BGR2HLS);
    Vec3b v = hls.at<Vec3b>(0,0);
    HLS_cv_to_user(v, Hv, Lv, Sv);

    Vec4f c = rgb2cmyk(Vec3b((int)round(Bv),(int)round(Gv),(int)round(Rv)));
    Cv=c[0]*100.0; Mv=c[1]*100.0; Yv=c[2]*100.0; Kv=c[3]*100.0;

    int px = paletteRect.x + (int)round((Hv/360.0)*(paletteRect.width-1));
    int py = paletteRect.y + (int)round((1.0 - Sv/100.0)*(paletteRect.height-1));
    paletteMarker = {px,py};
    if (redraw) imshow(WIN, canvas);
}
static void sync_from_hls(bool redraw=true){
    Mat hls(1,1,CV_8UC3, Scalar(0,0,0)), bgr;
    hls.at<Vec3b>(0,0) = HLS_user_to_cv(Hv, Lv, Sv);
    cvtColor(hls, bgr, COLOR_HLS2BGR);
    Vec3b p=bgr.at<Vec3b>(0,0); Bv=p[0]; Gv=p[1]; Rv=p[2];

    Vec4f c = rgb2cmyk(Vec3b((int)round(Bv),(int)round(Gv),(int)round(Rv)));
    Cv=c[0]*100.0; Mv=c[1]*100.0; Yv=c[2]*100.0; Kv=c[3]*100.0;

    int px = paletteRect.x + (int)round((Hv/360.0)*(paletteRect.width-1));
    int py = paletteRect.y + (int)round((1.0 - Sv/100.0)*(paletteRect.height-1));
    paletteMarker = {px,py};
    if (redraw) imshow(WIN, canvas);
}
static void sync_from_cmyk(bool redraw=true){
    Vec3b b = cmyk2rgb(Vec4f(Cv/100.f, Mv/100.f, Yv/100.f, Kv/100.f));
    Bv=b[0]; Gv=b[1]; Rv=b[2];

    Mat hls; cvtColor(bgr1x1(), hls, COLOR_BGR2HLS);
    Vec3b v = hls.at<Vec3b>(0,0);
    HLS_cv_to_user(v, Hv, Lv, Sv);

    int px = paletteRect.x + (int)round((Hv/360.0)*(paletteRect.width-1));
    int py = paletteRect.y + (int)round((1.0 - Sv/100.0)*(paletteRect.height-1));
    paletteMarker = {px,py};
    if (redraw) imshow(WIN, canvas);
}

// ======= Палитра с кэшем =======
static void drawHSMap(Mat& img){
    int Lcv = (int)round(clamp(Lv,0.0,100.0)*255.0/100.0);
    if (hsCache.empty() || hsCacheLcv!=Lcv){
        Mat hls(paletteRect.height, paletteRect.width, CV_8UC3);
        for(int y=0;y<paletteRect.height;++y){
            Vec3b* row=hls.ptr<Vec3b>(y);
            double Sp=100.0*(1.0-(double)y/(paletteRect.height-1));
            int Scv=(int)round(Sp*255.0/100.0);
            for(int x=0;x<paletteRect.width;++x){
                double Hd=360.0*((double)x/(paletteRect.width-1));
                int Hcv=(int)round(fmod(Hd,360.0)/2.0);
                row[x]=Vec3b((uchar)Hcv,(uchar)Lcv,(uchar)Scv);
            }
        }
        Mat bgr; cvtColor(hls,bgr,COLOR_HLS2BGR);
        hsCache=std::move(bgr); hsCacheLcv=Lcv;
    }
    hsCache.copyTo(img(paletteRect));
    rectangle(img,paletteRect,Scalar(20,20,20),2,LINE_AA);
    circle(img,paletteMarker,8,Scalar(255,255,255),2,LINE_AA);
    circle(img,paletteMarker,8,Scalar(0,0,0),1,LINE_AA);
}
static void setHSFromPoint(Point p){
    p.x=clamp(p.x,paletteRect.x,paletteRect.x+paletteRect.width-1);
    p.y=clamp(p.y,paletteRect.y,paletteRect.y+paletteRect.height-1);
    paletteMarker=p;
    double nx=(p.x-paletteRect.x)/(double)(paletteRect.width-1);
    double ny=(p.y-paletteRect.y)/(double)(paletteRect.height-1);
    Hv=nx*360.0; Sv=(1.0-ny)*100.0;
    sync_from_hls(false);
}

// ======= Геометрия =======
static const int BOX_W=100, BOX_H=32, SL_W=230, SL_H=18, GAP_X=12, ROW_H=56;
static Rect makeBox(int x,int y){ return Rect(x,y,BOX_W,BOX_H); }
static Rect sliderRightOf(const Rect& b){ return Rect(b.x+b.width+GAP_X,b.y+(BOX_H-SL_H)/2,SL_W,SL_H); }

static void buildFields(){
    fields.clear();
    rgbPanel={60,470,BOX_W+GAP_X+SL_W+40,3*ROW_H+40};
    cmykPanel={460,470,BOX_W+GAP_X+SL_W+40,4*ROW_H+40};
    hlsPanel={860,470,BOX_W+GAP_X+SL_W+40,3*ROW_H+40};
    auto rowY=[&](int base,int i){return base+40+i*ROW_H;};
    Rect b;
    b=makeBox(rgbPanel.x+20,rowY(rgbPanel.y,0));fields.emplace_back("R",b,0,255,&Rv,0);fields.back().slider=sliderRightOf(b);
    b=makeBox(rgbPanel.x+20,rowY(rgbPanel.y,1));fields.emplace_back("G",b,0,255,&Gv,0);fields.back().slider=sliderRightOf(b);
    b=makeBox(rgbPanel.x+20,rowY(rgbPanel.y,2));fields.emplace_back("B",b,0,255,&Bv,0);fields.back().slider=sliderRightOf(b);
    b=makeBox(cmykPanel.x+20,rowY(cmykPanel.y,0));fields.emplace_back("C",b,0,100,&Cv,1);fields.back().slider=sliderRightOf(b);
    b=makeBox(cmykPanel.x+20,rowY(cmykPanel.y,1));fields.emplace_back("M",b,0,100,&Mv,1);fields.back().slider=sliderRightOf(b);
    b=makeBox(cmykPanel.x+20,rowY(cmykPanel.y,2));fields.emplace_back("Y",b,0,100,&Yv,1);fields.back().slider=sliderRightOf(b);
    b=makeBox(cmykPanel.x+20,rowY(cmykPanel.y,3));fields.emplace_back("K",b,0,100,&Kv,1);fields.back().slider=sliderRightOf(b);
    b=makeBox(hlsPanel.x+20,rowY(hlsPanel.y,0));fields.emplace_back("H",b,0,360,&Hv,2);fields.back().slider=sliderRightOf(b);
    b=makeBox(hlsPanel.x+20,rowY(hlsPanel.y,1));fields.emplace_back("L",b,0,100,&Lv,2);fields.back().slider=sliderRightOf(b);
    b=makeBox(hlsPanel.x+20,rowY(hlsPanel.y,2));fields.emplace_back("S",b,0,100,&Sv,2);fields.back().slider=sliderRightOf(b);
}

// ======= Отрисовка =======
static void render(){
    canvas.setTo(Scalar((int)round(Bv),(int)round(Gv),(int)round(Rv)));
    rectangle(canvas,Rect(0,0,canvas.cols,120),Scalar(35,35,35),FILLED);
    ostringstream l1,l2;
    l1<<"RGB: "<<(int)round(Rv)<<","<<(int)round(Gv)<<","<<(int)round(Bv)
      <<"    HLS: "<<fixed<<setprecision(2)<<Hv<<","<<Lv<<"%,"<<Sv<<"%";
    l2<<"CMYK: "<<fixed<<setprecision(2)<<Cv<<","<<Mv<<","<<Yv<<","<<Kv;
    putTextShadow(canvas,l1.str(),{20,48},1.2,Scalar(240,240,240));
    putTextShadow(canvas,l2.str(),{20,92},1.2,Scalar(240,240,240));
    drawHSMap(canvas);
    putSmall(canvas,"Palette H(x) vs S(y) | Lightness L = "+to_string((int)round(Lv))+" %",{paletteRect.x,paletteRect.y-10});
    drawPanelBg(canvas,rgbPanel);drawPanelBg(canvas,cmykPanel);drawPanelBg(canvas,hlsPanel);
    putSmall(canvas,"RGB",{rgbPanel.x+10,rgbPanel.y+24});
    putSmall(canvas,"CMYK",{cmykPanel.x+10,cmykPanel.y+24});
    putSmall(canvas,"HLS",{hlsPanel.x+10,hlsPanel.y+24});
    for(size_t i=0;i<fields.size();++i){
        Field &f=fields[i];
        putSmall(canvas,f.label+":",{f.box.x-22,f.box.y+BOX_H-6});
        Scalar bg=(editState==EditState::Editing&&(int)i==editIndex)?Scalar(180,210,240):Scalar(30,30,30);
        rectangle(canvas,f.box,bg,FILLED);
        rectangle(canvas,f.box,Scalar(120,120,120),1,LINE_AA);
        ostringstream val;
        if(editState==EditState::Editing&&(int)i==editIndex) val<<inputBuf;
        else{
            bool frac=(f.group!=0);
            val<<fixed<<setprecision(frac?2:0)<<(*f.value);
            if(f.group==2){ if(f.label!="H") val<<"%"; }
            else if(f.group==1){ val<<'%'; }
        }
        putSmall(canvas,val.str(),{f.box.x+8,f.box.y+BOX_H-6});
        drawSlider(canvas,f);
    }
    imshow(WIN,canvas);
}

// ======= Ввод =======
static void commitEdit(){
    if(editState!=EditState::Editing||editIndex<0||editIndex>=(int)fields.size()){editState=EditState::None;return;}
    Field &f=fields[editIndex];
    if(!inputBuf.empty()){
        try{
            string s=inputBuf;
            s.erase(remove_if(s.begin(),s.end(),[](unsigned char c){return !(isdigit(c)||c=='-'||c=='+'||c=='.'||c==',');}),s.end());
            replace(s.begin(),s.end(),',','.');
            double v=stod(s);
            v=clamp(v,f.minv,f.maxv);
            if(fabs(v-*f.value)>1e-9){
                *f.value=v;
                if(f.group==0)sync_from_rgb(false);
                else if(f.group==1)sync_from_cmyk(false);
                else sync_from_hls(false);
            }
        }catch(...){}
    }
    inputBuf.clear();editState=EditState::None;render();
}

// ======= Мышь =======
static void on_mouse(int event,int x,int y,int,void*){
    Point p(x,y);
    if(event==EVENT_LBUTTONDOWN){
        if(paletteRect.contains(p)){draggingPalette=true;setHSFromPoint(p);render();return;}
        for(size_t i=0;i<fields.size();++i){
            if(fields[i].slider.contains(p)){
                editState=EditState::DragSlider;editIndex=(int)i;dragCounter=0;
                Field &f=fields[i];
                double t=(p.x-f.slider.x)/(double)(f.slider.width-1);
                t=clamp(t,0.0,1.0);
                *f.value=f.minv+t*(f.maxv-f.minv);
                if(f.group==0)sync_from_rgb(false);
                else if(f.group==1)sync_from_cmyk(false);
                else sync_from_hls(false);
                render();return;
            }
        }
        editState=EditState::None;editIndex=-1;
        for(size_t i=0;i<fields.size();++i){
            if(fields[i].box.contains(p)){editState=EditState::Editing;editIndex=(int)i;inputBuf.clear();render();return;}
        }
        render();return;
    }
    else if(event==EVENT_MOUSEMOVE){
        if(draggingPalette){
            if(++dragCounter%3==0){setHSFromPoint(p);render();}
            return;
        }
        if(editState==EditState::DragSlider&&editIndex>=0){
            Field &f=fields[editIndex];
            double t=(p.x-f.slider.x)/(double)(f.slider.width-1);
            t=clamp(t,0.0,1.0);
            double newVal=f.minv+t*(f.maxv-f.minv);
            if(fabs(newVal-*f.value)<1e-3)return;
            *f.value=newVal;
            if(++dragCounter%3==0){
                if(f.group==0)sync_from_rgb(false);
                else if(f.group==1)sync_from_cmyk(false);
                else sync_from_hls(false);
                render();
            }
            return;
        }
    }
    else if(event==EVENT_LBUTTONUP){
        draggingPalette=false;dragCounter=0;
        if(editState==EditState::DragSlider){editState=EditState::None;}
    }
}

// ======= MAIN =======
int main(){
    namedWindow(WIN);
    setMouseCallback(WIN,on_mouse);
    buildFields();
    sync_from_rgb(false);
    render();
    for(;;){
        int k=waitKey(1);
        if(k<0)continue;
        if(editState==EditState::Editing){
            if(k==27){editState=EditState::None;inputBuf.clear();render();}
            else if(k==13||k==10){commitEdit();}
            else if(k==8||k==127){if(!inputBuf.empty())inputBuf.pop_back();render();}
            else{
                if((k>='0'&&k<='9')||k=='.'||k==','||k=='-'||k=='+'){
                    char ch=(k==',')?'.':(char)k;inputBuf.push_back(ch);render();
                }
            }
        }else{
            if(k==27)break;
        }
    }
    destroyAllWindows();
    return 0;
}
