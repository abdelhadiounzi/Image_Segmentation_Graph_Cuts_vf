# Segmentation d'images par Graph Cuts

Notebook pédagogique : **`Image_Segmentation_Graph_Cuts_vf.ipynb`**

Ce notebook explique pas à pas le principe du **Graph Cuts** pour la segmentation d'images (séparation premier plan / arrière-plan), l'implémente à partir de zéro, puis le compare à d'autres approches : GrabCut, Normalized Cuts et une hybridation CNN + Graph Cuts. L'image de test est une photo de pomme rouge.

---

## Objectifs

- Comprendre comment une image devient un **graphe** (pixels = nœuds, source *S* et puits *T*).
- Maîtriser la **fonction d'énergie** `E(A) = λ·R(A) + B(A)` (terme régional + terme de frontière).
- Calculer la **coupe minimale** et vérifier le théorème **max-flow / min-cut**.
- Mesurer l'influence du paramètre **λ**.
- Comparer cinq méthodes de segmentation avec des métriques quantitatives (IoU, Dice, précision, rappel) et des temps de calcul.

---

## Contenu du notebook

| Section | Sujet | Description |
|---|---|---|
| 1 | Idée générale | N-links, T-links, coupe minimale, formulation de l'énergie globale |
| 2 | Construction du graphe | Chargement de l'image (40×40, niveaux de gris), graines premier plan / arrière-plan, fonction `build_graph`, visualisation d'un sous-graphe avec capacités |
| 3 | Coupe minimale | Fonction `segment` basée sur `networkx.minimum_cut`, masque et contour de segmentation |
| 4 | Influence de λ | Comparaison pour λ ∈ {0.05, 0.3, 1.0, 5.0} |
| 5 | Lien avec max-flow | Vérification numérique : flux maximal = coupe minimale |
| 6 | GrabCut (OpenCV) | Initialisation par rectangle, modèles de mélanges gaussiens (GMM) |
| 7 | Normalized Cuts | Superpixels SLIC + graphe d'adjacence de régions + `cut_normalized` (scikit-image), puis comparaison des trois premières approches |
| 8 | Graph Cuts + CNN | Petit CNN entièrement convolutif (convolutions dilatées) entraîné **uniquement sur les graines**, dont la sortie fournit les T-links ; solveur PyMaxflow ; étude de λ |
| 9 | Comparaison globale | Cinq méthodes comparées à une référence approchée (saturation + Otsu), cartes d'erreur, scores, temps de calcul et tableau qualitatif |

### Méthodes comparées

1. **Min-Cut standard** (pixel, niveaux de gris, graines)
2. **GrabCut** (GMM + rectangle)
3. **Normalized Cuts** (spectral, sur superpixels)
4. **CNN seul** (entraîné sur les graines)
5. **CNN + Graph Cuts** (probabilités du CNN comme terme régional, contraste couleur comme terme de frontière)

---

## Prérequis

- Python 3.8 ou supérieur
- Jupyter Notebook, JupyterLab ou Google Colab
- Une **connexion Internet** : l'image de la pomme est téléchargée depuis Wikimedia Commons à l'exécution (`apple.jpg`)

### Bibliothèques

| Bibliothèque | Usage |
|---|---|
| `numpy`, `scipy`, `pandas` | Calcul numérique, morphologie, tableaux de résultats |
| `matplotlib` | Visualisations |
| `networkx` | Graphe, `minimum_cut`, `maximum_flow_value` |
| `Pillow` | Chargement et redimensionnement des images |
| `opencv-python` | GrabCut |
| `scikit-image` | SLIC, graphe d'adjacence de régions, Normalized Cuts, Otsu |
| `torch` | CNN (entraînement et inférence, CPU ou GPU) |
| `PyMaxflow` | Solveur max-flow rapide (installé automatiquement par le notebook s'il est absent) |

### Installation

```bash
pip install numpy scipy pandas matplotlib networkx pillow opencv-python scikit-image torch PyMaxflow jupyter
```

---

## Utilisation

```bash
jupyter notebook Image_Segmentation_Graph_Cuts_vf.ipynb
```

Puis exécutez les cellules **dans l'ordre**, de haut en bas (`Run All`).

> **Important :** les sections se réutilisent mutuellement (`img`, `fg_seeds`, `bg_seeds`, `mask`, `img_color`, `im_bgr`, `fg_gc`, `segmented_ncut`, etc.). Exécuter une cellule isolément, ou dans le désordre, provoquera des erreurs ou des résultats incohérents.

### Temps d'exécution indicatif

- Sections 1 à 7 : rapides (le graphe `networkx` est volontairement limité à 40×40 pixels).
- Section 8 : entraînement du CNN (300 itérations sur une image 128 px), de quelques secondes à quelques dizaines de secondes sur CPU.
- Section 9 : ré-exécute chaque méthode pour mesurer les temps, la section Normalized Cuts étant la plus coûteuse.

---

## Paramètres à expérimenter

| Paramètre | Où | Effet |
|---|---|---|
| `lam` (λ) | `build_graph`, `segment`, `graphcut_from_probs` | Poids du terme de frontière : faible = suit les probabilités locales, élevé = frontières plus lisses |
| `sigma` (σ) | idem | Sensibilité aux contours dans le calcul de `B_pq` |
| `fg_seeds`, `bg_seeds` | Section 2 | Positions des graines (grille 40×40) |
| `rect` | Section 6 | Rectangle d'initialisation de GrabCut |
| `n_segments`, `compactness`, `thresh` | Section 7 | Superpixels SLIC et seuil de Normalized Cuts |
| `WORK`, `epochs`, `LAM_CNN` | Section 8 | Résolution de travail, durée d'entraînement et λ du modèle hybride |

Pour tester une autre image, modifiez l'URL dans la section 2 (ou chargez un fichier local) et **adaptez les graines** à la nouvelle image.

---

## Limites et remarques

- Il n'existe **pas de vérité terrain** pour l'image de test. La section 9 construit une **référence approchée** (seuillage d'Otsu sur la saturation, remplissage de trous, ouverture morphologique, plus grande composante connexe). Les scores servent à **classer les méthodes entre elles**, pas à fournir une évaluation absolue. Pour une évaluation rigoureuse, remplacez `ref_mask` par un masque dessiné à la main.
- Normalized Cuts est non supervisé et produit plusieurs régions : elles sont binarisées selon la proximité de leur couleur moyenne avec celle des graines.
- Le CNN est entraîné sur une seule image avec très peu de pixels supervisés : il illustre l'idée d'hybridation, pas une solution généralisable. La fonction `graphcut_from_probs` accepte n'importe quelle carte de probabilité (par exemple la sortie d'un U-Net ou d'un DeepLab pré-entraîné).
- Les résultats peuvent varier légèrement selon la machine (graines aléatoires fixées, mais différences CPU/GPU possibles).

---

## Références

- Boykov, Y. & Jolly, M.-P. (2001). *Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images.* ICCV.
- Rother, C., Kolmogorov, V. & Blake, A. (2004). *"GrabCut": Interactive Foreground Extraction using Iterated Graph Cuts.* SIGGRAPH.
- Shi, J. & Malik, J. (2000). *Normalized Cuts and Image Segmentation.* IEEE TPAMI.
- Boykov, Y. & Kolmogorov, V. (2004). *An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision.* IEEE TPAMI.
