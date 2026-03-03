# Package for Analyzing and Visualizing Win Sets

*Warning* Package is under active development and upgrades. The current version has been released for public testing and comment but is expected to undergo significant changes in the near future. 

The code should run on the standard python libraries. 

## figure_generator.py
In this module is the draw_win_set() command. It is the current workhorse engine for drawing win sets. It takes a number of parameters, but the most important is a list of tuples representing the (x,y) coordinate pairs for each voter's ideal point. Additional optional parameters allow for the following (non-exhaustive list):
-  Changing colors,
-  Adding a win set centroid (the average of the win set in both dimensions),
-  One or more reference points
-  Controlling indifference curve size (for "thick" indifference curves),
-  Adding reservation utility to create alienated voters,
-  Adding optional labels optionally in latex math format,

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Suggested Citation 
If you use this package to create visuals for your projects, please include the following citation (or equivalent):

Baugus, N. (2026). *winning_win_sets* (Version 1.0.0) [Computer software]. GitHub.

```bibtex
@software{baugus2026_winning_win_sets,
	author  = {Baugus, Nathanael},
	title   = {winning\_win\_sets},
	year    = {2026},
	version = {1.0.0},
	note    = {Computer software},
	url     = {https://github.com/Baugus/winning_win_sets}
}
```