

python setup.py build_ext --inplace


注释掉```lanms/__init__.py```中此两行
```python
# if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
#     raise RuntimeError('Cannot compile lanms: {}'.format(BASE_DIR))
```