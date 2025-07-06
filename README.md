# AAAI-25-MRFF
Code for aaai-25 paper "[Multifaceted User Modeling in Recommendation: A Federated Foundation Models Approach](https://ojs.aaai.org/index.php/AAAI/article/view/33440)"

## Abatract
Multifaceted user modeling aims to uncover fine-grained patterns and learn representations from user data, revealing their diverse interests and characteristics, such as profile, preference, and personality. Recent studies on foundation model-based recommendation have emphasized the Transformer architecture's remarkable ability to capture complex, non-linear user-item interaction relationships. This paper aims to advance foundation model-based recommender systems by introducing enhancements to multifaceted user modeling capabilities. We propose a novel Transformer layer designed specifically for recommendation, using the self-attention mechanism to capture sequential user-item interaction patterns. Specifically, we design a group gating network to identify user groups, enabling hierarchical discovery across different layers, thereby capturing the multifaceted nature of user interests through multiple Transformer layers. Furthermore, to broaden the data scope and further enhance multifaceted user modeling, we extend the framework to a federated setting, enabling the use of private datasets while ensuring privacy. Experimental validations on benchmark datasets demonstrate the superior performance of our proposed method.

![](https://github.com/Zhangcx19/AAAI-25-MRFF/blob/main/framework.png
)

**Figure:**
The model architecture of proposed MRFF.


## Citation
If you find this project helpful, please consider to cite the following paper:

```
@inproceedings{zhang2025multifaceted,
  title={Multifaceted user modeling in recommendation: A federated foundation models approach},
  author={Zhang, Chunxu and Long, Guodong and Guo, Hongkuan and Liu, Zhaojie and Zhou, Guorui and Zhang, Zijian and Liu, Yang and Yang, Bo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={12},
  pages={13197--13205},
  year={2025}
}
```
