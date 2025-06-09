# Differentiable Computation with Awkward Array and JAX

[![Talk](https://img.shields.io/badge/MODE25-talk-blue?logo=github&logoColor=white&color=blue)](https://indi.to/8WZTS)

Modern scientific computing often involves nested and variable-length data structures, which pose challenges for automatic differentiation (AD). Awkward Array is a library for manipulating irregular data and its integration with JAX enables forward and reverse mode AD on irregular data. Several Python libraries, such as PyTorch, TensorFlow, and Zarr, offer variations of ragged data structures, but differentiating through their ragged types remains impossible or problematic. Awkward's JAX backend allows users to differentiate nested and variable-length data structures without compromising readability, ease of use, and performance.

This talk presents the current status of the Awkward Array's JAX backend, highlighting its implementation using JAX's pytrees, tracing mechanisms, and compatibility with JAX's AD system. We discuss the coverage of Awkward Array's automatic differentiation support, strategies for differentiable programming with nested data, and challenges encountered in extending JAX's API to support non-rectilinear array structures. Finally, we outline future development directions, including keeping up with JAX's evolving AD ecosystem, improved interoperability with ML frameworks, and potential applications in physics and beyond

## Stuck somewhere? Reach out!

- If something is not working the way it should, or if you want to request a new feature, create a [new issue](https://github.com/scikit-hep/awkward/issues) on GitHub.
- To discuss something related to vector, use the [discussions](https://github.com/scikit-hep/awkward/discussions/) tab on GitHub.
- Have a look at vector's [releases](https://github.com/scikit-hep/awkward/releases) to stay up-to-date!

## Cite awkward

If you use `awkward`'s `jax` backend in your work, please cite this presentation.

More broadly, if you use `awkward` in your work, please cite it using the following metadata -

```bib
@software{Pivarski_Awkward_Array_2018,
author = {Pivarski, Jim and Osborne, Ianna and Ifrim, Ioana and Schreiner, Henry and Hollands, Angus and Biswas, Anish and Das, Pratyush and Roy Choudhury, Santam and Smith, Nicholas and Goyal, Manasvi},
doi = {10.5281/zenodo.4341376},
month = oct,
title = {{Awkward Array}},
year = {2018}
}
```
