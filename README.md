# GeckoBotModel
Modeling Tools for predict and simulate the behavior of the gecko-inspired soft robot 


## Kinematic model for predicting quasi-static configuration for given input

![Example](https://github.com/larslevity/GeckoBotModel/blob/master/Pics/kinematic_model/principle_sketch.PNG)


For structures whose elements consist of bending actuators, the use of a PCC model is reasonable. In structures in which closed kinematic chains occur, however, this assumption is inaccurate. Due to constraint conditions, the elements do not necessarily deform in a circular arc. In addition, the permissible range for the joint coordinates is restricted depending on the working point. Nevertheless, In order to be able to represent such structures with a PCC model, the model is extended by virtual lengths. Figure 1 illustrates this principle. A single bending actuator is shown in (a). For a given bending angle and arc length, a discrete point results for its tip. However, if a deviation in angle and length is allowed, its tip may be located in the light red area. This is
particularly helpful when considering a structure as shown in (b). It consists of three bending actuators which are joined at 90 ◦ to each other. The structure is rigidly clamped on both sides. For given reference angles (i.e., bending angles which the elements should adopt), it is now possible to calculate how the overall structure will deform. That is, circular arcs are arranged in such a way that their end points correspond to those of the clamping. Thereby, the circular arcs should have a minimum deviation from the reference angles and the nominal lengths. This does not necessarily reflect the actual behavior (compare grey dashed lines in Fig. 1(b)), but the position of the endpoints can be approximated sufficiently well.

![Example](https://github.com/larslevity/GeckoBotModel/blob/master/Pics/kinematic_model/model_new.png)



## Pathplanning algorithm based on gait law, derived from kinematic model





![Example](https://github.com/larslevity/GeckoBotModel/blob/master/Pics/path_planning/model_cost.png)



|![Example](https://github.com/larslevity/GeckoBotModel/blob/master/Pics/path_planning/eval_cost_fun.png)|![Example](https://github.com/larslevity/GeckoBotModel/blob/master/Pics/path_planning/corresponding_gait.png)|
|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|

Evaluation of the cost function for curve (left Figure); right Figure shows the resulting simulated optimal gait, corresponding to the minimum distance (marked by a yellow circle).