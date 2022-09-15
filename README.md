# Algorithms-and-Machine-Learning
## Content
Linear regression:
- Closed form fit
- Gradient descent approach
- L1, L2 and Elastic net regularization

Algorithms:
- Minimum cost to connect grid coordinates

## [Pre-commit](https://pre-commit.com/)
See https://pre-commit.com for more information \
See https://pre-commit.com/hooks.html for more hooks
```
pip install pre-commit

pre-commit sample-config > .pre-commit-config.yaml

# On Windows (encoding error)
pre-commit sample-config | out-file .pre-commit-config.yaml -encoding utf8

pre-commit install
```

You can use `pre-commit autoupdate` to update the dependencies to the latest version automatically.

## [mypy]()
Variable typing: \
https://github.com/python/mypy/blob/master/docs/source/type_inference_and_annotations.rst
