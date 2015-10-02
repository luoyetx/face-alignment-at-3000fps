#ifndef RANDOMFOREST_HPP
#define RANDOMFOREST_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include "lbf/common.hpp"

namespace lbf {

class RandomTree {
public:
    RandomTree();
    ~RandomTree();
    //RandomTree(const RandomTree &other);
    //RandomTree &operator=(const RandomTree &other);

public:
    void Init(int landmark_id, int depth);
    void Train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes, \
               std::vector<cv::Mat> &delta_shapes, cv::Mat &mean_shape, std::vector<int> &index, int stage);
    void SplitNode(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes, \
                   cv::Mat &delta_shapes, cv::Mat &mean_shape, std::vector<int> &root, int idx, int stage);

    void Read(FILE *fd);
    void Write(FILE *fd);

public:
    int depth;
    int nodes_n;
    int landmark_id;
    cv::Mat_<double> feats;
    std::vector<int> thresholds;
};

class RandomForest {
public:
    RandomForest();
    ~RandomForest();
    //RandomForest(const RandomForest &other);
    //RandomForest &operator=(const RandomForest &other);

public:
    void Init(int landmark_n, int trees_n, int tree_depth);
    void Train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, std::vector<cv::Mat> &current_shapes, \
               std::vector<BBox> &bboxes, std::vector<cv::Mat> &delta_shapes, cv::Mat &mean_shape, int stage);
    cv::Mat GenerateLBF(cv::Mat &img, cv::Mat &current_shape, BBox &bbox, cv::Mat &mean_shape);

    void Read(FILE *fd);
    void Write(FILE *fd);

public:
    int landmark_n;
    int trees_n, tree_depth;
    std::vector<std::vector<RandomTree> > random_trees;
};

} // namespace lbf

#endif // RANDOMFOREST_HPP
