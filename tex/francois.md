# Non-dimensionalisation

## Cahn Hilliard equation

Start from Cahn-Hilliard equation:
$$
\begin{align*}
\frac{\partial c}{\partial t} &= \nabla M \nabla \mu\\
&=\nabla M \nabla \left[Aw'(c) - \kappa \Delta c\right]
\end{align*}
$$
$c$ is a non-dimensionalised parameter, $[A]=E/L^3$, $[\kappa]=E/L^3 \times L^2$, $[M]=[D]/(E/L^3)$.

Let $\overline{M} = MA$ and $\overline{\kappa}=\kappa/A$.

Let $x=x^*\ l_c$, $\overline{M}=\overline{M}^*\ \overline{M}_0$, $t=t^*\ \tau$

This leads to:

$$
\begin{align*}
\frac{1}{\tau} &\frac{\partial c}{\partial t^*} = \frac{\overline{M}_0}{l_c^2} \nabla^* \overline{M}^* \nabla^* \left[w'(c) - \frac{\overline{\kappa}}{l_c^2} \Delta^* c\right]\\
&\frac{\partial c}{\partial t^*} = \nabla^* \overline{M}^* \nabla^* \left[\frac{\overline{M}_0\ \tau}{l_c^2} w'(c) -  \frac{\overline{M}_0\ \tau}{l_c^2}\frac{\overline{\kappa}}{l_c^2} \Delta^* c\right]\\
\implies &\frac{\partial c}{\partial t^*} = \nabla^* \overline{M}^* \nabla^* \left[A^* w'(c) - \kappa^* \Delta^* c\right]
\end{align*}
$$
Thus one defines the following non-dimensionalised parameters:
$A^*=\overline{M}_0\ \tau / l^2_c$ and $\kappa^*=\overline{M}_0\ \tau / l^2_c \cdot \overline{\kappa}/l^2_c$

If one takes $\overline{M}_0=3.3633\ m^2/s$, $l_c=10\ nm$ and consider $\overline{M}^*=1$ then:
$$\tau = l^2_c / M \simeq 30\ ns$$

If one also considers $\kappa^*=1$, expected $t_d$ for a single free-standing NW of diameter $D=120\ nm$ i.e. $D = 12 \ l_c$ is $O(1000\  \tau)$.
