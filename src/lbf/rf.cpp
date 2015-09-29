#include <cmath>
#include <cstdio>
#include <cassert>
#include "lbf/rf.hpp"

using namespace cv;
using namespace std;

namespace lbf {

#define SIMILARITY_TRANSFORM(x, y, scale, rotate) do {            \
        double x_tmp = scale * (rotate(0, 0)*x + rotate(0, 1)*y); \
        double y_tmp = scale * (rotate(1, 0)*x + rotate(1, 1)*y); \
        x = x_tmp; y = y_tmp;                                     \
    } while(0)

RandomTree::RandomTree() {}
RandomTree::~RandomTree() {}
RandomTree::RandomTree(const RandomTree &other) {}
RandomTree &RandomTree::operator=(const RandomTree &other) {
    if (this == &other) return *this;
    return *this;
}

void RandomTree::Init(int landmark_id, int depth) {
    this->landmark_id = landmark_id;
    this->depth = depth;
    nodes_n = 1 << depth;
    feats = Mat::zeros(nodes_n, 4, CV_64FC1);
    thresholds.resize(nodes_n);
}

void RandomTree::Train(vector<Mat> &imgs, vector<Mat> &current_shapes, vector<BBox> &bboxes, \
    vector<Mat> &delta_shapes, Mat &mean_shape, vector<int> &index, int stage) {
    Mat_<double> delta_shapes_(delta_shapes.size(), 2);
    for (int i = 0; i < delta_shapes.size(); i++) {
        delta_shapes_(i, 0) = delta_shapes[i].at<double>(landmark_id, 0);
        delta_shapes_(i, 1) = delta_shapes[i].at<double>(landmark_id, 1);
    }
    SplitNode(imgs, current_shapes, bboxes, delta_shapes_, mean_shape, index, 1, stage);
}

void RandomTree::SplitNode(vector<Mat> &imgs, vector<Mat> &current_shapes, vector<BBox> &bboxes, \
    Mat &delta_shapes, Mat &mean_shape, vector<int> &root, int idx, int stage) {
    Config &config = Config::GetInstance();
    int N = root.size();
    if (N == 0) {
        thresholds[idx] = 0;
        feats.row(idx).setTo(0);
        vector<int> left, right;
        // split left and right child in DFS
        if (2 * idx < feats.rows / 2)
            SplitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, left, 2 * idx, stage);
        if (2 * idx + 1 < feats.rows / 2)
            SplitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, right, 2 * idx + 1, stage);
        return;
    }

    int feats_m = config.feats_m[stage];
    double radius_m = config.radius_m[stage];
    Mat_<double> candidate_feats(feats_m, 4);
    RNG rng(getTickCount());
    // generate feature pool
    for (int i = 0; i < feats_m; i++) {
        double x1, y1, x2, y2;
        x1 = rng.uniform(-1., 1.); y1 = rng.uniform(-1., 1.);
        x2 = rng.uniform(-1., 1.); y2 = rng.uniform(-1., 1.);
        if (x1*x1 + y1*y1 > 1. || x2*x2 + y2*y2 > 1.) {
            i--;
            continue;
        }
        candidate_feats[i][0] = x1 * radius_m;
        candidate_feats[i][1] = y1 * radius_m;
        candidate_feats[i][2] = x2 * radius_m;
        candidate_feats[i][3] = y2 * radius_m;
    }
    // calc features
    Mat_<int> densities(feats_m, N);
    for (int i = 0; i < N; i++) {
        double scale;
        Mat_<double> rotate;
        const Mat_<double> &current_shape = (Mat_<double>)current_shapes[root[i]];
        BBox &bbox = bboxes[root[i]];
        Mat &img = imgs[root[i]];
        calcSimilarityTransform(bbox.Project(current_shape), mean_shape, scale, rotate);
        for (int j = 0; j < feats_m; j++) {
            double x1 = candidate_feats(j, 0);
            double y1 = candidate_feats(j, 1);
            double x2 = candidate_feats(j, 2);
            double y2 = candidate_feats(j, 3);
            SIMILARITY_TRANSFORM(x1, y1, scale, rotate);
            SIMILARITY_TRANSFORM(x2, y2, scale, rotate);

            x1 = x1*bbox.x_scale + current_shape(landmark_id, 0);
            y1 = y1*bbox.y_scale + current_shape(landmark_id, 1);
            x2 = x2*bbox.x_scale + current_shape(landmark_id, 0);
            y2 = y2*bbox.y_scale + current_shape(landmark_id, 1);
            x1 = max(0., min(img.cols - 1., x1)); y1 = max(0., min(img.rows - 1., y1));
            x2 = max(0., min(img.cols - 1., x2)); y2 = max(0., min(img.rows - 1., y2));
            densities(j, i) = (int)img.at<uchar>(int(y1), int(x1)) - (int)img.at<uchar>(int(y2), int(x2));
        }
    }
    Mat_<int> densities_sorted;
    cv::sort(densities, densities_sorted, SORT_EVERY_ROW + SORT_ASCENDING);
    //select a feat which reduces maximum variance
    double variance_all = (calcVariance(delta_shapes.col(0)) + calcVariance(delta_shapes.col(1)))*N;
    double variance_reduce_max = 0;
    int threshold = 0;
    int feat_id = 0;
    vector<double> left_x, left_y, right_x, right_y;
    left_x.reserve(N); left_y.reserve(N);
    right_x.reserve(N); right_y.reserve(N);
    for (int j = 0; j < feats_m; j++) {
        left_x.clear(); left_y.clear();
        right_x.clear(); right_y.clear();
        int threshold_ = densities_sorted(j, (int)(N*rng.uniform(0.05, 0.95)));
        for (int i = 0; i < N; i++) {
            if (densities(j, i) < threshold_) {
                left_x.push_back(delta_shapes.at<double>(root[i], 0));
                left_y.push_back(delta_shapes.at<double>(root[i], 1));
            }
            else {
                right_x.push_back(delta_shapes.at<double>(root[i], 0));
                right_y.push_back(delta_shapes.at<double>(root[i], 1));
            }
        }
        double variance_ = (calcVariance(left_x) + calcVariance(left_y))*left_x.size() + \
            (calcVariance(right_x) + calcVariance(right_y))*right_x.size();
        double variance_reduce = variance_all - variance_;
        if (variance_reduce > variance_reduce_max) {
            variance_reduce_max = variance_reduce;
            threshold = threshold_;
            feat_id = j;
        }
    }
    thresholds[idx] = threshold;
    feats(idx, 0) = candidate_feats(feat_id, 0); feats(idx, 1) = candidate_feats(feat_id, 1);
    feats(idx, 2) = candidate_feats(feat_id, 2); feats(idx, 3) = candidate_feats(feat_id, 3);
    // generate left and right child
    vector<int> left, right;
    left.reserve(N);
    right.reserve(N);
    for (int i = 0; i < N; i++) {
        if (densities(feat_id, i) < threshold) left.push_back(root[i]);
        else right.push_back(root[i]);
    }
    // split left and right child in DFS
    if (2 * idx < feats.rows / 2)
        SplitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, left, 2 * idx, stage);
    if (2 * idx + 1 < feats.rows / 2)
        SplitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, right, 2 * idx + 1, stage);
}

void RandomTree::Read(FILE *fd) {
    Config &config = Config::GetInstance();
    // initialize
    Init(0, config.tree_depth);
    for (int i = 1; i < nodes_n / 2; i++) {
        fread(feats.ptr<double>(i), sizeof(double), 4, fd);
        fread(&thresholds[i], sizeof(int), 1, fd);
    }
}

void RandomTree::Write(FILE *fd) {
    for (int i = 1; i < nodes_n / 2; i++) {
        fwrite(feats.ptr<double>(i), sizeof(double), 4, fd);
        fwrite(&thresholds[i], sizeof(int), 1, fd);
    }
}


RandomForest::RandomForest() {}
RandomForest::~RandomForest() {}
RandomForest::RandomForest(const RandomForest &other) {}
RandomForest &RandomForest::operator=(const RandomForest &other) {
    if (this == &other) return *this;
    return *this;
}

void RandomForest::Init(int landmark_n, int trees_n, int tree_depth) {
    random_trees.resize(landmark_n);
    for (int i = 0; i < landmark_n; i++) {
        random_trees[i].resize(trees_n);
        for (int j = 0; j < trees_n; j++) random_trees[i][j].Init(i, tree_depth);
    }
    this->trees_n = trees_n;
    this->landmark_n = landmark_n;
    this->tree_depth = tree_depth;
}

void RandomForest::Train(vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<Mat> &current_shapes, \
    vector<BBox> &bboxes, vector<Mat> &delta_shapes, Mat &mean_shape, int stage) {
    int N = imgs.size();
    double overlap_ratio = Config::GetInstance().bagging_overlap;
    int Q = int(N / ((1. - overlap_ratio) * trees_n));

#pragma omp parallel for
    for (int i = 0; i < landmark_n; i++) {
    TIMER_BEGIN
        vector<int> root;
        for (int j = 0; j < trees_n; j++) {
            int start = max(0, int(floor(j*Q - j*Q*overlap_ratio)));
            int end = min(int(start + Q + 1), N);
            int L = end - start;
            root.resize(L);
            for (int k = 0; k < L; k++) root[k] = start + k;
            random_trees[i][j].Train(imgs, current_shapes, bboxes, delta_shapes, mean_shape, root, stage);
        }
        LOG("Train %2dth landmark Done, it costs %.4lf s", i, TIMER_NOW);
    TIMER_END
    }
}

Mat RandomForest::GenerateLBF(Mat &img, Mat &current_shape, BBox &bbox, Mat &mean_shape) {
    Mat_<int> lbf(1, landmark_n*trees_n);
    double scale;
    Mat_<double> rotate;
    calcSimilarityTransform(bbox.Project(current_shape), mean_shape, scale, rotate);

    int base = 1 << (tree_depth - 1);

    //#pragma omp parallel for num_threads(2)
    for (int i = 0; i < landmark_n; i++) {
        for (int j = 0; j < trees_n; j++) {
            RandomTree &tree = random_trees[i][j];
            int code = 0;
            int idx = 1;
            for (int k = 1; k < tree.depth; k++) {
                double x1 = tree.feats(idx, 0);
                double y1 = tree.feats(idx, 1);
                double x2 = tree.feats(idx, 2);
                double y2 = tree.feats(idx, 3);
                SIMILARITY_TRANSFORM(x1, y1, scale, rotate);
                SIMILARITY_TRANSFORM(x2, y2, scale, rotate);

                x1 = x1*bbox.x_scale + current_shape.at<double>(i, 0);
                y1 = y1*bbox.y_scale + current_shape.at<double>(i, 1);
                x2 = x2*bbox.x_scale + current_shape.at<double>(i, 0);
                y2 = y2*bbox.y_scale + current_shape.at<double>(i, 1);
                x1 = max(0., min(img.cols - 1., x1)); y1 = max(0., min(img.rows - 1., y1));
                x2 = max(0., min(img.cols - 1., x2)); y2 = max(0., min(img.rows - 1., y2));
                int density = img.at<uchar>(int(y1), int(x1)) - img.at<uchar>(int(y2), int(x2));
                code <<= 1;
                if (density < tree.thresholds[idx]) {
                    idx = 2 * idx;
                }
                else {
                    code += 1;
                    idx = 2 * idx + 1;
                }
            }
            lbf(i*trees_n + j) = (i*trees_n + j)*base + code;
        }
    }
    return lbf;
}


void RandomForest::Read(FILE *fd) {
    Config &config = Config::GetInstance();
    // initialize
    Init(config.landmark_n, config.tree_n, config.tree_depth);
    for (int i = 0; i < landmark_n; i++) {
        for (int j = 0; j < trees_n; j++) {
            random_trees[i][j].Read(fd);
        }
    }
}

void RandomForest::Write(FILE *fd) {
    for (int i = 0; i < landmark_n; i++) {
        for (int j = 0; j < trees_n; j++) {
            random_trees[i][j].Write(fd);
        }
    }
}

} // namespace lbf
