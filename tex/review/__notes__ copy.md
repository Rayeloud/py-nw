# Planning

## Aknowledgments (tout écrire)

## 1. Introduction (40%)
  - Context (truc joli)
  - Objectives (mieux reformuler et etoffer)
  - Outline (Compléter et reformuler)

## 3. Free-standing nanowire
### Non-dimensional analysis (fait)
### Geometrical considerations (80%)
  - Explication sur la cross-section pentagonale à étoffer
  - Conséquence énergétique (notion de structure crystalline, anistropie énergétique)
  - Problèmes de lié à la modélisation de structure polycrystalline (coût computationnel)
  - Motivation d'une approche isotropique avec une forme pentagonale
  - Comparaison de l'énergie libre entre l'approximation pentagonale et approximation circulaire
### Results and discussion (70%)
  Caractérisation de l'instabilité morphologique en regardant 2 grandeurs, temps de cassure (prop au taux de croissances des instabilités), longueure d'onde de l'instabilité (Comparaison à la théorie Plateau-Rayleigh des fluides)
  - Single nanowire
    - Dépendence $R/L^*$ (initial radius)
      - Motivation (comparaison avec la littérature)
      - Graphs
    - Dépendence $R/L$ (aspect-ratio)
      - Motivation (Notion d'ultra-high aspect-ratio nanowire et range d'aspect ratio (dans une solution, nanowire de longueurs différentes))
      - Graphs
    - Full functional dependencies of $t_d$ and $\lambda$
  - Junction of two free-standing nanowire
    Motivation générale
    - Dependence $R_1/L^*$
    - Dependence $R_1/R_2$, relative aspect ratio
      - Motivation (Solution de NW, différents radii dans la même solution -> impact sur les jonctions résultantes et la stabilité du réseau)
      - Graphs
    - Dependence $\theta$, relative orientation
        - Motivation (Différentes techniques de déposition mènent à différentes orientation relative entre les fils)
        - Graphs
    - Full functional dependencies of $t_d$
  - Discussion (simple discussion et rappel des résultats + comparaison à Mullins et McCallum (cas $\theta_B=\pi$))
## 4. Nanowire on substrate
### Smooth boundary method (80%)
- Motivation (étoffer)
- Mathematical derivation (fait)
- Modified Cahn Hilliard equation
  - Motivation
  - Introduction de la notion d'angle de contact (à étoffer)
- Validation
  - Motivation, discussion (fait)
  - Graphs (à refaire)
### Model of the configuration (70%)
- Contact angle condition
  - Discussion de la pertinence de la condition d'angle de contact dans le cadre des nanofils métalliques (à étoffer)
  - Analyse des images SEM de nanofils sphéroidiser (~70 nanodots) (fait)
    - Graphs fréquence (à rajouter)
    - Hypothèse, angle de contact initial maintenu (à étoffer)
- Junction configuration (nanowire in contact and nanowire ~free-standing -> SEM image) (à étoffer)
- Geometry of computational domain
  - Graph de domain order parameter + composition field (nanowire voxelisé) (à rajouter)
  - approximation circulaire, considération sur l'énergie de surface (à étoffer)
### Comparative analysis (50%)
  Motivation générale (à étoffer)
  - Single nanowire
    - Results
    - Full functional dependency of $t_d$ and $\lambda$ for varying $\theta_B$
  - Junction nanowire
    - Results
    - Full functional dependency of $t_d$ for varying $\theta_B$
  - Discussion (Comparaison avec théorie de McCallum pour les fils seuls, etc)

## 5. Conclusion (10%)
- Conclusion générale
  - Rappel du context et de la question de recherche
  - Réponse à la question de recherche
  - Rappel du fil conducteur (ce qui a été fait dans le chap1->chap4)
  - Résultats de chaque chaps
  - Réponse à la question
- Perspectives
  - Modèle plus complet capturant l'anisotropie du fils (modèle KKS (couplage Cahn-Hilliard Allen-Cahn), terme quartic de l'expansion de l'énergie libre), utilisation de framework plus puissant (framework MOOSE de résolution de PDE non linéraire)