Statement of the problem: 

$$
    -\nabla^2 u = f, \quad u(\Omega) = 0
$$

Where $\Omega$ is the unit square, and $f$ is randomly generated function of the form:

$$
    f(x,y) = \sum_{i=1}^{n} a_i \sin(k_{x,i} \pi x + \phi_{x,i}) \sin(k_{y,i} \pi y + \phi_{y,i})
$$

Where:
- $a_i \in [-1, 1]$ is the amplitude
- $k_{x,i}, k_{y,i} \in [1, 4]$ are the wave numbers
- $\phi_{x,i}, \phi_{y,i} \in [0, 2\pi]$ are the phases

Problem is solved numerically using a 5 point stencil finite difference approximation of the laplacian:

$$
-\nabla^2 u_{i,j} \approx \frac{u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1} + 4u_{i,j}}{h^2}
$$

Which yields the final approximation for each interior point $(i,j)$:

$$
-u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1} + 4u_{i,j} = h^2f_{i,j}
$$

And points on the boundary:

$$
u_{i,j} = 0
$$