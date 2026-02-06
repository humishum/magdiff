# Deciding how to represent position and orientations 

_this is an informal doc..._ 

We need a way to represent position and orientation in 3D space for magnets(and in some way fields), that is 
1) easy to represent and instantiate 
2) gradient friendly 

Ulimately we need to represent 6 DOF(3 position and 3 orientation)

There's probably a few ways to do this: 
1. Position(xyz) and Euler Angles 
    Dependent on order of rotations
    Wrapping euler angles will probably be hard(impossible??) since the gradient will be massive. i.e. if we're reducing angles -179, and that wraps to 180, that's a huge gradient.

2. Position(xyz) and Quaternion representation 
    No gimbal lock! 
    the really big con I see here is that we need to normalize the quaternion every step to maintain ∣∣q∣∣=1 . we should try this out, but this seems "expensive"
    Quaternion is 4 parameters! 

3.  Position(xyz) and Matrix Vector Representation 

    this seems like the simplest option, we have a 6 dof system, and have 6 params to constrain. this matrix doens't need to be normalized or constrained. 

    could always construct the 4x4 from this easily too? 

    θ=π might be a problem but this doesn't seem too bad of an issue??? we shal see 



There's also the options of using the 4x4 SE(3) matrix directly(but the whole bottom row is kind of a waste for us?), or Lie algebra(of which I'm not very familiar with atm, but maybe worth revisiting later!)