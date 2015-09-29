#include <ctime>
#include <cstdio>
#include <cassert>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lbf/lbf.hpp"

#include <iostream>

using namespace cv;
using namespace std;
using namespace lbf;

void parseTxt(string &txt, vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes) {
    Config &config = Config::GetInstance();
    FILE *fd = fopen(txt.c_str(), "r");
    assert(fd);
    int N;
    int landmark_n = config.landmark_n;
    fscanf(fd, "%d", &N);
    imgs.resize(N);
    gt_shapes.resize(N);
    bboxes.resize(N);
    char img_path[256];
    double bbox[4];
    vector<double> x(landmark_n), y(landmark_n);
    for (int i = 0; i < N; i++) {
        fscanf(fd, "%s", img_path);
        for (int j = 0; j < 4; j++) {
            fscanf(fd, "%lf", &bbox[j]);
        }
        for (int j = 0; j < landmark_n; j++) {
            fscanf(fd, "%lf%lf", &x[j], &y[j]);
        }
        Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
        // crop img
        double x_min, y_min, x_max, y_max;
        x_min = *min_element(x.begin(), x.end());
        x_max = *max_element(x.begin(), x.end());
        y_min = *min_element(y.begin(), y.end());
        y_max = *max_element(y.begin(), y.end());
        x_min = max(0., x_min - bbox[2] / 2);
        x_max = min(img.cols - 1., x_max + bbox[2] / 2);
        y_min = max(0., y_min - bbox[3] / 2);
        y_max = min(img.rows - 1., y_max + bbox[3] / 2);
        double x_, y_, w_, h_;
        x_ = x_min; y_ = y_min;
        w_ = x_max - x_min; h_ = y_max - y_min;
        bboxes[i] = BBox(bbox[0] - x_, bbox[1] - y_, bbox[2], bbox[3]);
        gt_shapes[i] = Mat::zeros(landmark_n, 2, CV_64FC1);
        for (int j = 0; j < landmark_n; j++) {
            gt_shapes[i].at<double>(j, 0) = x[j] - x_;
            gt_shapes[i].at<double>(j, 1) = y[j] - y_;
        }
        Rect roi(x_, y_, w_, h_);
        imgs[i] = img(roi).clone();
    }
    fclose(fd);
}


void data_augmentation(vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes) {
    int N = imgs.size();
    imgs.reserve(2 * N);
    gt_shapes.reserve(2 * N);
    bboxes.reserve(2 * N);
    for (int i = 0; i < N; i++) {
        Mat img_flipped;
        Mat_<double> gt_shape_flipped(gt_shapes[i].size());
        flip(imgs[i], img_flipped, 1);
        int w = img_flipped.cols - 1;
        int h = img_flipped.rows - 1;
        for (int k = 0; k < gt_shapes[i].rows; k++) {
            gt_shape_flipped(k, 0) = w - gt_shapes[i].at<double>(k, 0);
            gt_shape_flipped(k, 1) = gt_shapes[i].at<double>(k, 1);
        }
        int x_b, y_b, w_b, h_b;
        x_b = w - bboxes[i].x - bboxes[i].width;
        y_b = bboxes[i].y;
        w_b = bboxes[i].width;
        h_b = bboxes[i].height;
        BBox bbox_flipped(x_b, y_b, w_b, h_b);

        imgs.push_back(img_flipped);
        gt_shapes.push_back(gt_shape_flipped);
        bboxes.push_back(bbox_flipped);

        //Mat tmp = drawShapeInImage(imgs[i], gt_shapes[i], bboxes[i]);
        //imshow("img", tmp);
        //waitKey(0);
        //cout << w << endl;
        //cout << gt_shapes[i] << endl;
        //cout << gt_shape_flipped << endl;
        //tmp = drawShapeInImage(img_flipped, gt_shape_flipped, bbox_flipped);
        //imshow("img", tmp);
        //waitKey(0);
    }
    // landmark id need swap
    Config &config = Config::GetInstance();

#define SWAP(shape, i, j) do { \
        double tmp = shape.at<double>(i-1, 0); \
        shape.at<double>(i-1, 0) = shape.at<double>(j-1, 0); \
        shape.at<double>(j-1, 0) = tmp; \
        tmp =  shape.at<double>(i-1, 1); \
        shape.at<double>(i-1, 1) = shape.at<double>(j-1, 1); \
        shape.at<double>(j-1, 1) = tmp; \
    } while(0)

    if (config.landmark_n == 29) {
        for (int i = N; i < gt_shapes.size(); i++) {
            SWAP(gt_shapes[i], 1, 2);
            SWAP(gt_shapes[i], 3, 4);
            SWAP(gt_shapes[i], 5, 7);
            SWAP(gt_shapes[i], 6, 8);
            SWAP(gt_shapes[i], 13, 15);
            SWAP(gt_shapes[i], 9, 10);
            SWAP(gt_shapes[i], 11, 12);
            SWAP(gt_shapes[i], 17, 18);
            SWAP(gt_shapes[i], 14, 16);
            SWAP(gt_shapes[i], 18, 20);
            SWAP(gt_shapes[i], 23, 24);
        }
    }
    else if (config.landmark_n == 68) {
        for (int i = N; i < gt_shapes.size(); i++) {
            for (int k = 1; k <= 8; k++) SWAP(gt_shapes[i], k, 18 - k);
            for (int k = 18; k <= 22; k++) SWAP(gt_shapes[i], k, 45 - k);
            for (int k = 37; k <= 40; k++) SWAP(gt_shapes[i], k, 83 - k);
            SWAP(gt_shapes[i], 42, 47);
            SWAP(gt_shapes[i], 41, 48);
            SWAP(gt_shapes[i], 32, 36);
            SWAP(gt_shapes[i], 33, 35);
            for (int k = 49; k <= 51; k++) SWAP(gt_shapes[i], k, 104 - k);
            SWAP(gt_shapes[i], 60, 56);
            SWAP(gt_shapes[i], 59, 57);
            SWAP(gt_shapes[i], 61, 65);
            SWAP(gt_shapes[i], 62, 64);
            SWAP(gt_shapes[i], 68, 66);
        }
    }
    else {
        LOG("Wrang Landmark_n, it must be 29 or 68");
    }

#undef SWAP

}


int train(int start_from) {
    Config &config = Config::GetInstance();
    LOG("Load train data from %s", config.dataset.c_str());
    string txt = config.dataset + "/train.txt";
    vector<Mat> imgs_, gt_shapes_;
    vector<BBox> bboxes_;
    parseTxt(txt, imgs_, gt_shapes_, bboxes_);

    LOG("Data Augmentation");
    data_augmentation(imgs_, gt_shapes_, bboxes_);
    Mat mean_shape = getMeanShape(gt_shapes_, bboxes_);

    int N = imgs_.size();
    int L = N*config.initShape_n;
    vector<Mat> imgs(L), gt_shapes(L), current_shapes(L);
    vector<BBox> bboxes(L);
    RNG rng(getTickCount());
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < config.initShape_n; j++) {
            int idx = i*config.initShape_n + j;
            int k = 0;
            do {
                //if (i < (N / 2)) k = rng.uniform(0, N / 2);
                //else k = rng.uniform(N / 2, N);
                k = rng.uniform(0, N);
            } while (k == i);
            imgs[idx] = imgs_[i];
            gt_shapes[idx] = gt_shapes_[i];
            bboxes[idx] = bboxes_[i];
            current_shapes[idx] = bboxes_[i].ReProject(bboxes_[k].Project(gt_shapes_[k]));
        }
    }
    // random shuffle
    std::srand(std::time(0));
    std::random_shuffle(imgs.begin(), imgs.end());
    std::srand(std::time(0));
    std::random_shuffle(gt_shapes.begin(), gt_shapes.end());
    std::srand(std::time(0));
    std::random_shuffle(bboxes.begin(), bboxes.end());
    std::srand(std::time(0));
    std::random_shuffle(current_shapes.begin(), current_shapes.end());

    LbfCascador lbf_cascador;
    lbf_cascador.Init(config.stages_n);
    if (start_from > 0) {
        lbf_cascador.ResumeTrainModel(start_from);
    }
    // Train
    TIMER_BEGIN
        lbf_cascador.Train(imgs, gt_shapes, current_shapes, bboxes, mean_shape, start_from);
        LOG("Train Model Down, cost %.4lf s", TIMER_NOW);
    TIMER_END

    // Save
    FILE *fd = fopen(config.saved_file_name.c_str(), "wb");
    assert(fd);
    lbf_cascador.Write(fd);
    fclose(fd);

    return 0;
}
