#include <cstdio>
#include <cstdlib>
#include "liblinear/linear.h"
#include "lbf/lbf.hpp"

using namespace cv;
using namespace std;

namespace lbf {

LbfCascador::LbfCascador() {}
LbfCascador::~LbfCascador() {}
LbfCascador::LbfCascador(const LbfCascador &other) {}
LbfCascador &LbfCascador::operator=(const LbfCascador &other) {
    if (this == &other) return *this;
    return *this;
}

void LbfCascador::Init(int stages_n) {
    Config &config = Config::GetInstance();
    this->stages_n = stages_n;
    this->landmark_n = config.landmark_n;
    random_forests.resize(stages_n);
    for (int i = 0; i < stages_n; i++) random_forests[i].Init(config.landmark_n, config.tree_n, config.tree_depth);
    mean_shape.create(config.landmark_n, 2, CV_64FC1);
    gl_regression_weights.resize(stages_n);
    int F = config.landmark_n * config.tree_n * (1 << (config.tree_depth - 1));
    for (int i = 0; i < stages_n; i++) {
        gl_regression_weights[i].create(2 * config.landmark_n, F, CV_64FC1);
    }
}

void LbfCascador::Train(vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<Mat> &current_shapes, \
    vector<BBox> &bboxes, Mat &mean_shape_, int start_from) {
    assert(start_from >= 0 && start_from < stages_n);
    mean_shape = mean_shape_;
    int N = imgs.size();
    int landmark_n = gt_shapes[0].rows;

    if (start_from > 0) {
        // resume current_shapes
        for (int i = 0; i < N; i++) {
            Mat &current_shape = current_shapes[i];
            BBox &bbox = bboxes[i];
            Mat &img = imgs[i];
            double scale;
            Mat rotate;

            for (int k = 0; k < start_from; k++) {
                // generate lbf
                Mat lbf = random_forests[k].GenerateLBF(img, current_shape, bbox, mean_shape);
                // update current_shapes
                Mat delta_shape = GlobalRegressionPredict(lbf, k);
                delta_shape = delta_shape.reshape(0, landmark_n);
                calcSimilarityTransform(bbox.Project(current_shape), mean_shape, scale, rotate);
                current_shape = bbox.ReProject(bbox.Project(current_shape) + scale * delta_shape * rotate.t());
            }
        }
        // calc mean error
        double e = calcMeanError(gt_shapes, current_shapes);
        LOG("Resume Done with Error = %lf", e);
    }

    for (int k = start_from; k < stages_n; k++) {
        vector<Mat> delta_shapes = getDeltaShapes(gt_shapes, current_shapes, bboxes, mean_shape);
        // train random forest
        LOG("start train random forest of %dth stage", k);
        TIMER_BEGIN
            random_forests[k].Train(imgs, gt_shapes, current_shapes, bboxes, delta_shapes, mean_shape, k);
        LOG("end of train random forest of %dth stage, costs %.4lf s", k, TIMER_NOW);
        TIMER_END

            // generate lbf of every train data
            vector<Mat> lbfs;
        lbfs.resize(N);
        for (int i = 0; i < N; i++) {
            lbfs[i] = random_forests[k].GenerateLBF(imgs[i], current_shapes[i], bboxes[i], mean_shape);
        }
        // global regression
        LOG("start train global regression of %dth stage", k);
        TIMER_BEGIN
            GlobalRegressionTrain(lbfs, delta_shapes, k);
        LOG("end of train global regression of %dth stage, costs %.4lf s", k, TIMER_NOW);
        TIMER_END
            // update current_shapes
            double scale;
        Mat rotate;
        for (int i = 0; i < N; i++) {
            Mat delta_shape = GlobalRegressionPredict(lbfs[i], k);
            calcSimilarityTransform(bboxes[i].Project(current_shapes[i]), mean_shape, scale, rotate);
            current_shapes[i] = bboxes[i].ReProject(bboxes[i].Project(current_shapes[i]) + scale * delta_shape * rotate.t());
        }
        // calc mean error
        double e = calcMeanError(gt_shapes, current_shapes);
        LOG("Train %dth stage Done with Error = %lf", k, e);

        // dump current mode
        DumpTrainModel(k);
    }
}

// Dump current model, end_to shold lie in [0, stage_n)
void LbfCascador::DumpTrainModel(int stage) {
    assert(stage >= 0 && stage < stages_n);
    LOG("Dump model of stage %d", stage);
    char buff[50];
    sprintf(buff, "../models/tmp_stage_%d.model", stage);

    FILE *fout = fopen(buff, "wb");
    assert(fout);

    double *ptr;
    random_forests[stage].Write(fout);
    for (int i = 0; i < 2 * landmark_n; i++) {
        ptr = gl_regression_weights[stage].ptr<double>(i);
        fwrite(ptr, sizeof(double), gl_regression_weights[stage].cols, fout);
    }

    fclose(fout);
}

// Resume training from `start_from`, start_from shold lie in [1, stage_n)
void LbfCascador::ResumeTrainModel(int start_from) {
    assert(start_from >= 1 && start_from < stages_n);
    LOG("Resuming training from stage %d", start_from);

    char buff[50];
    FILE *fin;
    double *ptr;
    for (int k = 0; k < start_from; k++) {
        sprintf(buff, "../models/tmp_stage_%d.model", k);
        fin = fopen(buff, "rb");
        assert(fin);

        random_forests[k].Read(fin);
        for (int i = 0; i < 2 * landmark_n; i++) {
            ptr = gl_regression_weights[k].ptr<double>(i);
            fread(ptr, sizeof(double), gl_regression_weights[k].cols, fin);
        }
    }
}

void LbfCascador::Test(vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes) {
    int N = imgs.size();
    vector<Mat> current_shapes(N);
    TIMER_BEGIN
        for (int i = 0; i < N; i++) {
        current_shapes[i] = bboxes[i].ReProject(mean_shape);
        }
    double e = calcMeanError(gt_shapes, current_shapes);
    LOG("initial error = %.6lf", e);
    double scale;
    Mat rotate;
    for (int k = 0; k < stages_n; k++) {
        for (int i = 0; i < N; i++) {
            // generate lbf
            Mat lbf = random_forests[k].GenerateLBF(imgs[i], current_shapes[i], bboxes[i], mean_shape);
            // update current_shapes
            Mat delta_shape = GlobalRegressionPredict(lbf, k);
            delta_shape = delta_shape.reshape(0, landmark_n);
            current_shapes[i] = bboxes[i].Project(current_shapes[i]);
            calcSimilarityTransform(current_shapes[i], mean_shape, scale, rotate);
            current_shapes[i] = bboxes[i].ReProject(current_shapes[i] + scale * delta_shape * rotate.t());
        }
        e = calcMeanError(gt_shapes, current_shapes);
        LOG("stage %dth error = %.6lf", k, e);
    }
    LOG("Test %d images costs %.4lf s with %.2lf fps", N, TIMER_NOW, N / TIMER_NOW);
    TIMER_END
}

// Global Regression to predict delta shape with LBF
void LbfCascador::GlobalRegressionTrain(vector<Mat> &lbfs, vector<Mat> &delta_shapes, int stage) {
    Config &config = Config::GetInstance();
    int N = lbfs.size();
    int M = lbfs[0].cols;
    int F = config.landmark_n*config.tree_n*(1 << (config.tree_depth - 1));
    int landmark_n = delta_shapes[0].rows;
    // prepare linear regression params X and Y
    struct feature_node **X = (struct feature_node **)malloc(N * sizeof(struct feature_node *));
    double **Y = (double **)malloc(landmark_n * 2 * sizeof(double *));
    for (int i = 0; i < N; i++) {
        X[i] = (struct feature_node *)malloc((M + 1) * sizeof(struct feature_node));
        for (int j = 0; j < M; j++) {
            X[i][j].index = lbfs[i].at<int>(0, j) + 1; // index starts from 1
            X[i][j].value = 1;
        }
        X[i][M].index = X[i][M].value = -1;
    }
    for (int i = 0; i < landmark_n; i++) {
        Y[2 * i] = (double *)malloc(N*sizeof(double));
        Y[2 * i + 1] = (double *)malloc(N*sizeof(double));
        for (int j = 0; j < N; j++) {
            Y[2 * i][j] = delta_shapes[j].at<double>(i, 0);
            Y[2 * i + 1][j] = delta_shapes[j].at<double>(i, 1);
        }
    }
    // train every landmark
    struct problem prob;
    struct parameter param;
    prob.l = N;
    prob.n = F;
    prob.x = X;
    prob.bias = -1;
    param.solver_type = L2R_L2LOSS_SVR_DUAL;
    param.C = 1. / N;
    param.p = 0;
    param.eps = 0.00001;

    Mat_<double> weight(2 * landmark_n, F);

#pragma omp parallel for
    for (int i = 0; i < landmark_n; i++) {

#define FREE_MODEL(model)	\
    free(model->w);			\
    free(model->label);		\
    free(model)

        LOG("train %2dth landmark", i);
        struct problem prob_ = prob;
        prob_.y = Y[2 * i];
        check_parameter(&prob_, &param);
        struct model *model = train(&prob_, &param);
        for (int j = 0; j < F; j++) weight(2 * i, j) = get_decfun_coef(model, j + 1, 0);
        FREE_MODEL(model);

        prob_.y = Y[2 * i + 1];
        check_parameter(&prob_, &param);
        model = train(&prob_, &param);
        for (int j = 0; j < F; j++) weight(2 * i + 1, j) = get_decfun_coef(model, j + 1, 0);
        FREE_MODEL(model);

#undef FREE_MODEL

    }

    gl_regression_weights[stage] = weight;

    // free
    for (int i = 0; i < N; i++) free(X[i]);
    for (int i = 0; i < 2 * landmark_n; i++) free(Y[i]);
    free(X);
    free(Y);
}

// TODO: speed up
Mat LbfCascador::GlobalRegressionPredict(const Mat &lbf, int stage) {
    const Mat_<double> &weight = (Mat_<double>)gl_regression_weights[stage];
    Mat_<double> delta_shape(weight.rows / 2, 2);
    const double *w_ptr = NULL;
    const int *lbf_ptr = lbf.ptr<int>(0);

    //#pragma omp parallel for num_threads(2) private(w_ptr)
    for (int i = 0; i < delta_shape.rows; i++) {
        w_ptr = weight.ptr<double>(2 * i);
        double y = 0;
        for (int j = 0; j < lbf.cols; j++) y += w_ptr[lbf_ptr[j]];
        delta_shape(i, 0) = y;

        w_ptr = weight.ptr<double>(2 * i + 1);
        y = 0;
        for (int j = 0; j < lbf.cols; j++) y += w_ptr[lbf_ptr[j]];
        delta_shape(i, 1) = y;
    }
    return delta_shape;
}

Mat LbfCascador::Predict(Mat &img, BBox &bbox) {
    Mat current_shape = bbox.ReProject(mean_shape);
    double scale;
    Mat rotate;
    for (int k = 0; k < stages_n; k++) {
        // generate lbf
        Mat lbf = random_forests[k].GenerateLBF(img, current_shape, bbox, mean_shape);
        // update current_shapes
        Mat delta_shape = GlobalRegressionPredict(lbf, k);
        delta_shape = delta_shape.reshape(0, landmark_n);
        calcSimilarityTransform(bbox.Project(current_shape), mean_shape, scale, rotate);
        current_shape = bbox.ReProject(bbox.Project(current_shape) + scale * delta_shape * rotate.t());
    }
    return current_shape;
}

void LbfCascador::Read(FILE *fd) {
    Config &config = Config::GetInstance();
    // global parameters
    fread(&config.stages_n, sizeof(int), 1, fd);
    fread(&config.tree_n, sizeof(int), 1, fd);
    fread(&config.tree_depth, sizeof(int), 1, fd);
    fread(&config.landmark_n, sizeof(int), 1, fd);
    stages_n = config.stages_n;
    landmark_n = config.landmark_n;
    // initialize
    Init(stages_n);
    // mean_shape
    double *ptr = NULL;
    for (int i = 0; i < mean_shape.rows; i++) {
        ptr = mean_shape.ptr<double>(i);
        fread(ptr, sizeof(double), mean_shape.cols, fd);
    }
    // every stages
    for (int k = 0; k < stages_n; k++) {
        random_forests[k].Read(fd);
        for (int i = 0; i < 2 * config.landmark_n; i++) {
            ptr = gl_regression_weights[k].ptr<double>(i);
            fread(ptr, sizeof(double), gl_regression_weights[k].cols, fd);
        }
    }
}

void LbfCascador::Write(FILE *fd) {
    Config &config = Config::GetInstance();
    // global parameters
    fwrite(&config.stages_n, sizeof(int), 1, fd);
    fwrite(&config.tree_n, sizeof(int), 1, fd);
    fwrite(&config.tree_depth, sizeof(int), 1, fd);
    fwrite(&config.landmark_n, sizeof(int), 1, fd);
    // mean_shape
    double *ptr = NULL;
    for (int i = 0; i < mean_shape.rows; i++) {
        ptr = mean_shape.ptr<double>(i);
        fwrite(ptr, sizeof(double), mean_shape.cols, fd);
    }
    // every stages
    for (int k = 0; k < config.stages_n; k++) {
        LOG("Write %dth stage", k);
        random_forests[k].Write(fd);
        for (int i = 0; i < 2 * config.landmark_n; i++) {
            ptr = gl_regression_weights[k].ptr<double>(i);
            fwrite(ptr, sizeof(double), gl_regression_weights[k].cols, fd);
        }
    }
}

} // namespace lbf
