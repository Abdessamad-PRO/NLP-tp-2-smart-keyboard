# <img src="assets/icons/keyboard.svg" width="34" height="34" align="top">&nbsp;Lab 02 — Smart Keyboard V1

> **Un moteur de suggestions de mots façon clavier de téléphone, construit avec un modèle de langage n-gramme, évalué par la perplexité et la précision top-3, et présenté dans une interface Streamlit claire/sombre.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%E2%89%A5%201.30-FF4B4B?logo=streamlit&logoColor=white)
![Niveau](https://img.shields.io/badge/Niveau-d%C3%A9butant%20%E2%86%92%20interm%C3%A9diaire-6366f1)
![Données](https://img.shields.io/badge/Donn%C3%A9es-synth%C3%A9tiques-0ea5e9)

<p align="center">
  <img width="959" height="439" alt="Aperçu de l'application Smart Keyboard V1 (barre latérale, indicateurs et onglet Clavier)" src="https://github.com/user-attachments/assets/0acf1d34-7f4a-404c-aa51-368f5f2770ea" />
</p>

---

## Sommaire

1. [Objectif du TP](#objectif)
2. [Données](#donnees)
3. [Concepts et formules](#concepts)
4. [Structure du projet](#structure)
5. [Installation et lancement](#installation)
6. [Ce qui a été réalisé](#realise)
7. [Analyse des échecs](#echecs)
8. [Tour de l'interface Streamlit](#interface)
9. [Réponses aux questions de réflexion](#reflexion)
10. [Limites, vie privée et éthique](#limites)
11. [Pistes d'extension](#extensions)

---

<a id="objectif"></a>
## <img src="assets/icons/target.svg" width="26" height="26" align="top">&nbsp;1. Objectif du TP

Un **modèle de langage** attribue une probabilité au mot suivant, sachant ce qui a déjà été tapé :

$$P(\text{mot suivant} \mid \text{historique})$$

C'est l'idée la plus ancienne et la plus simple du NLP moderne, et c'est exactement la même quantité que les grands modèles neuronaux optimisent aujourd'hui.

**But final :** entraîner des modèles **unigramme, bigramme et trigramme**, les **lisser** pour qu'une séquence jamais vue n'obtienne pas une probabilité nulle, puis les comparer avec la **perplexité** et la **précision top-3** sur des documents mis de côté.

À l'issue du TP, on sait :

- calculer $P(w \mid \text{contexte})$ à partir de comptages et expliquer l'estimation du maximum de vraisemblance ;
- expliquer pourquoi un modèle non lissé échoue (probabilités nulles → perplexité infinie) et le corriger avec le **lissage add-k**, l'**interpolation** et le **back-off** ;
- calculer et interpréter la **perplexité** et la précision **top-k** ;
- séparer les données **par document** (et non par phrase) et expliquer la fuite que cela évite ;
- diagnostiquer les erreurs en listant les contextes jamais vus à l'entraînement.

### Pipeline

```
Corpus (60 documents)
      │
      ▼
Découpage PAR DOCUMENT (train / test)
      │
      ▼
Comptage des n-grammes (+ <s>, </s>, <unk>)
      │
      ▼
Lissage add-k · interpolation · back-off
      │
      ▼
Suggestions top-3 · perplexité · précision top-3
```

---

<a id="donnees"></a>
## <img src="assets/icons/database.svg" width="26" height="26" align="top">&nbsp;2. Données

| Élément | Valeur |
|---|---|
| Corpus | 1 200 phrases synthétiques réparties en **60 documents** |
| Taille | ≈ 10 000 tokens, **61 types de mots** |
| Génération | Grammaire à graine fixe (`python data/make_data.py`) → tout le monde obtient les mêmes données |
| Format | `data/corpus.csv` avec les colonnes `doc_id` et `sentence` |
| Découpage | Les **20 % derniers documents** forment le test : **960 phrases d'entraînement / 240 de test** |
| Tokens | Train : 8 098 · Test : 2 070 |

> **Pourquoi découper par document et pas par phrase ?**
> Les phrases d'un même document partagent des mots thématiques. Mettre certaines en train et d'autres en test laisserait le modèle « voir » le vocabulaire du test et rendrait les scores meilleurs qu'ils ne le sont vraiment (**fuite de données**).

---

<a id="concepts"></a>
## <img src="assets/icons/calculator.svg" width="26" height="26" align="top">&nbsp;3. Concepts et formules

### Symboles spéciaux

- `<s>` : début de phrase (ajouté $n-1$ fois pour un modèle d'ordre $n$) ;
- `</s>` : fin de phrase — le modèle doit aussi la **prédire** ;
- `<unk>` : mot inconnu. `clean()` y ramène tout mot hors vocabulaire pour que le modèle ne plante jamais.

### Trigramme avec lissage add-k

$$P(w \mid u\,v)=\frac{\text{count}(u\,v\,w)+k}{\text{count}(u\,v)+k\cdot V}$$

où $V$ est la taille du vocabulaire.

- $k = 0$ → un contexte inconnu donne une probabilité **0** : le modèle est fragile ;
- $k = 1$ → trop généreux : beaucoup de masse pour des séquences impossibles ;
- $k$ petit (≈ 0,01 à 0,1) → bon compromis.

### Perplexité

$$PP=\exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i \mid \text{contexte}_i)\right)$$

*Intuition : le modèle est aussi hésitant que s'il devait choisir uniformément parmi PP mots à chaque étape.* Plus c'est bas, mieux c'est. Un tirage uniforme sur ce vocabulaire donnerait environ **61**.

### Interpolation linéaire

$$P = \lambda_3 P_3 + \lambda_2 P_2 + \lambda_1 P_1 \qquad (\text{défaut : } 0{,}6 \;/\; 0{,}3 \;/\; 0{,}1)$$

### Back-off

On utilise le **plus grand n-gramme dont le contexte a déjà été vu** : trigramme → bigramme → unigramme.

### Précision top-k

Part des positions où le vrai mot suivant figure parmi les $k$ suggestions (ici $k=3$).

---

<a id="structure"></a>
## <img src="assets/icons/folder.svg" width="26" height="26" align="top">&nbsp;4. Structure du projet

```
02_smart_keyboard_v1/
├── app.py                  # Interface Streamlit (ce que nous avons réalisé)
├── requirements.txt        # matplotlib, streamlit
├── assets/
│   └── icons/              # icônes SVG utilisées par ce README
├── data/
│   ├── make_data.py        # génère corpus.csv (graine fixe)
│   └── corpus.csv
├── src/
│   ├── pipeline.py         # NGramModel, InterpolatedModel, split_by_document, top_k_accuracy…
│   └── evaluate.py         # évaluation complète, backoff_accuracy, balayage de k
└── results/
    ├── metrics.json
    ├── failures.txt
    └── ngram_results.png
```

> `app.py` **ne modifie aucun fichier du projet** : il utilise uniquement `src/pipeline.py` et `src/evaluate.py`.

---

<a id="installation"></a>
## <img src="assets/icons/rocket.svg" width="26" height="26" align="top">&nbsp;5. Installation et lancement

```bash
# 1. se placer dans le dossier du lab
cd 02_smart_keyboard_v1

# 2. créer un environnement isolé (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. installer les paquets
pip install -r requirements.txt  # = matplotlib streamlit

# 4. générer le jeu de données (graine fixe)
python data/make_data.py

# 5. lancer l'évaluation complète (imprime tous les chiffres du rapport)
python -m src.evaluate

# 6. lancer l'application web
streamlit run app.py
```

---

<a id="realise"></a>
## <img src="assets/icons/wrench.svg" width="26" height="26" align="top">&nbsp;6. Ce qui a été réalisé

### Partie A — Le moteur de langage (`src/pipeline.py`)

| Composant | Rôle |
|---|---|
| `split_by_document` | Découpage reproductible : les 20 % derniers documents = test |
| `NGramModel` | Comptage des n-grammes (`ctx_counts`, `ctx_totals`), `clean()`, `context()`, `prob()` avec add-k, `next_word_distribution()`, `top_k()`, `seen_context()`, `perplexity()` |
| `InterpolatedModel` | Mélange pondéré trigramme / bigramme / unigramme |
| `top_k_accuracy` | Précision top-k + liste des échecs |
| `backoff_accuracy` (`src/evaluate.py`) | Précision top-3 du back-off dur 3→2→1 |

### Partie B — L'évaluation (`python -m src.evaluate`)

- tableau comparatif des modèles (perplexité, part de tokens à probabilité 0, top-3) ;
- balayage du lissage $k$ pour le bigramme et le trigramme ;
- export de `results/metrics.json`, `results/failures.txt` et `results/ngram_results.png` ;
- repérage des contextes trigrammes jamais vus à l'entraînement.

### Partie C — L'application Streamlit (`app.py`)

- **Suggestions en direct** en tapant ou en cliquant sur un mot proposé, avec auto-complétion par préfixe ;
- **5 méthodes de prédiction** : back-off, trigramme, interpolé, bigramme, unigramme ;
- **Transparence** : explication des comptes réels et de la formule derrière chaque suggestion ;
- **Notation de phrase** : perplexité et « surprise » (en bits) mot par mot ;
- **Génération aléatoire** contrôlée par la température et la graine ;
- **Évaluation** : résultats du rapport + recalcul en direct avec les paramètres choisis ;
- **Robustesse** : un mot inconnu est signalé, traité comme `<unk>`, et l'application ne plante jamais ;
- **Design adaptatif** clair/sombre ;
- **Mises en cache** (`st.cache_data`, `st.cache_resource`) pour que les modèles ne soient entraînés qu'une fois par combinaison de paramètres.

---

<a id="echecs"></a>
## <img src="assets/icons/search.svg" width="26" height="26" align="top">&nbsp;7. Analyse des échecs

| Type d'échec | Exemple | Explication |
|---|---|---|
| **Ambiguïté en début de phrase** | `the` → *foxes* | Après « the », presque n'importe quel nom est possible ; 3 suggestions ne couvrent pas ~20 noms |
| **Contexte jamais vu** | `lazy soup` | Combinaison que la grammaire n'a jamais produite à l'entraînement → il faut **reculer vers un ordre inférieur** (back-off) |
| **Choix sémantique** | `eat the` → *cakes* | Plusieurs aliments sont équiprobables ; un n-gramme n'a aucune notion de sens |

9 positions de test ont un contexte trigramme absent de l'entraînement (ex. `brave foxes`, `brave girls`, `happy bread`, `hungry key`, `lazy soup`).

**Correctif proposé :** utiliser le **back-off** ou l'**interpolation** plutôt qu'un trigramme pur. C'est précisément ce que l'onglet *Comparer* de l'application met en évidence.

---

<a id="interface"></a>
## <img src="assets/icons/monitor.svg" width="26" height="26" align="top">&nbsp;8. Tour de l'interface Streamlit

### <img src="assets/icons/sliders.svg" width="20" height="20" align="top">&nbsp;Barre latérale

| Contrôle | Effet |
|---|---|
| **Méthode de prédiction** | Back-off 3→2→1 (par défaut), Trigramme, Interpolé 3/2/1, Bigramme, Unigramme |
| **Lissage k** | 0,001 · 0,01 · 0,1 (par défaut) · 0,5 · 1,0 |
| **Nombre de suggestions** | De 1 à 6 (3 par défaut) |
| **Utiliser tout le corpus** | Inclut le test dans l'entraînement (fuite) → l'évaluation en direct est désactivée |
| **Poids de l'interpolation** | λ₃ (0,6), λ₂ (0,3) ; λ₁ = 1 − λ₃ − λ₂ est calculé automatiquement |

### <img src="assets/icons/layout.svg" width="20" height="20" align="top">&nbsp;En-tête et indicateurs

Quatre cartes résument l'état courant : **Vocabulaire V**, **Phrases train**, **Méthode** et **Lissage k**.

---

### <img src="assets/icons/keyboard.svg" width="20" height="20" align="top">&nbsp;Onglet « Clavier »

Le cœur de l'application. On tape un début de phrase (ou on clique sur une suggestion) et le modèle propose les mots suivants.

- champ de saisie initialisé avec `the happy girls` ;
- case **« Je suis en train d'écrire un mot »** : active l'auto-complétion par préfixe, comme sur un vrai clavier ;
- bouton **Effacer** ;
- **boutons de suggestion** affichant le mot et sa probabilité ; le survol donne la valeur exacte `P = …` ;
- le badge indique le **modèle réellement utilisé** (ex. `3-gram`, ou `2-gram` en cas de repli) ;
- **histogramme** des 10 mots les plus probables ;
- encart **« Pourquoi ces suggestions ? »** : comptes réels $n(c,w)$ et $n(c)$, formule avec les valeurs numériques, tableau des probabilités ;
- **avertissements utiles** : mots hors vocabulaire, contexte jamais vu, préfixe sans correspondance, repli du back-off.

<p align="center">
  <img width="787" height="440" alt="Onglet Clavier : suggestions et histogramme des probabilités" src="https://github.com/user-attachments/assets/07bf019d-26ea-4abc-b4b9-83abbf9297c9" />
</p>

---

### <img src="assets/icons/balance.svg" width="20" height="20" align="top">&nbsp;Onglet « Comparer »

Un tableau qui met côte à côte les **5 méthodes** pour le texte courant : modèle réellement utilisé, contexte vu ou non à l'entraînement, et top-3 avec probabilités. On y voit qu'un trigramme sur un contexte inconnu donne des suggestions arbitraires, ce que le back-off corrige.

<p align="center">
  <img width="959" height="436" alt="Onglet Comparer : les cinq méthodes côte à côte" src="https://github.com/user-attachments/assets/a6449a13-fb05-4228-848c-df715083f5e3" />
</p>

---

### <img src="assets/icons/note.svg" width="20" height="20" align="top">&nbsp;Onglet « Phrase »

On saisit une phrase complète (terminée par `.` pour inclure la fin de phrase) et l'application calcule :

- la **perplexité de la phrase** ;
- un graphique de la **surprise par mot** (en bits, $-\log_2 p$) ;
- un tableau détaillé : position, mot, probabilité, surprise, mot inconnu ;
- le **mot le plus surprenant**.

---

### <img src="assets/icons/dice.svg" width="20" height="20" align="top">&nbsp;Onglet « Générer »

Génération aléatoire de phrases à partir du modèle choisi :

- **Température** (0,2 à 2,0) : < 1 prudent, > 1 plus aléatoire ;
- **Nombre de phrases** (1 à 10) ;
- **Graine aléatoire** pour des résultats reproductibles.

<p align="center">
  <img width="785" height="440" alt="Onglet Générer : phrases aléatoires, température et graine" src="https://github.com/user-attachments/assets/899285b3-29b8-4298-9132-bfbaca520565" />
</p>

---

### <img src="assets/icons/chart.svg" width="20" height="20" align="top">&nbsp;Onglet « Évaluation »

**Résultats du rapport** : tableau issu de `results/metrics.json` (perplexité, part de tokens à p = 0, top-3) et figure `ngram_results.png` (perplexité selon k, qualité des suggestions).

<p align="center">
  <img width="830" height="439" alt="Onglet Évaluation : tableau des résultats du rapport et graphiques" src="https://github.com/user-attachments/assets/c90ecb45-7790-4276-b02a-1e7cc107af60" />
</p>

**Analyse d'échecs** (`results/failures.txt`) et **évaluation en direct** : le bouton *Calculer maintenant* recalcule perplexité et top-3 avec les paramètres actuels (k, λ). Elle est désactivée quand « tout le corpus » est coché, pour éviter la fuite.

<p align="center">
  <img width="830" height="412" alt="Onglet Évaluation : analyse d'échecs et évaluation en direct" src="https://github.com/user-attachments/assets/e3bddf1d-923b-40fc-a748-6d058fdfb84f" />
</p>

---

### <img src="assets/icons/info.svg" width="20" height="20" align="top">&nbsp;Onglet « À propos »

Résumé du jeu de données, de la méthode d'évaluation honnête (découpage par document), des limites des n-grammes, de la vie privée, et rappel des paramètres actuels (V, k, λ, méthode, corpus utilisé).

---

<a id="reflexion"></a>
## <img src="assets/icons/lightbulb.svg" width="26" height="26" align="top">&nbsp;9. Réponses aux questions de réflexion

1. **Pourquoi la perplexité se dégrade entre bigramme (k=1) et trigramme (k=1), mais s'améliore avec k=0,1 ?**
   Un trigramme a beaucoup plus de contextes, chacun vu rarement. Avec k=1, la masse réservée $k\cdot V$ est énorme par rapport à de petits comptes $n(c)$ et dilue les bonnes prédictions. Avec k=0,1, cette masse gaspillée diminue et le trigramme tire parti de son contexte plus riche.
2. **Un 5-gramme aiderait-il sur 960 phrases ?**
   Probablement non : plus l'ordre augmente, plus les contextes sont rares (**dispersion des données**). La plupart des contextes de test ne seraient jamais vus et il faudrait reculer vers des ordres inférieurs.
3. **Pourquoi la perplexité n'est-elle pas comparable entre un modèle de mots et un modèle de caractères ?**
   Elle est calculée *par token* : prédire un caractère parmi un petit alphabet et prédire un mot parmi des milliers n'ont pas la même échelle ni la même unité.
4. **Comment utiliser l'application sans envoyer le texte à un serveur ?**
   En exécutant tout **en local** : les modèles sont entraînés sur la machine, rien n'est envoyé ni enregistré. C'est le cas ici.

---

<a id="limites"></a>
## <img src="assets/icons/alert.svg" width="26" height="26" align="top">&nbsp;10. Limites, vie privée et éthique

- **Données synthétiques et grammaticales par construction** : le texte réel est bien plus difficile (des perplexités de 100 et plus sont normales).
- **Aucun sens** : un n-gramme ne comprend pas la sémantique (« eat the » → plusieurs aliments équiprobables) et ne voit que $n-1$ mots de contexte.
- **Vie privée** : tout tourne en local, le texte tapé n'est ni envoyé ni enregistré. Un clavier entraîné sur des messages personnels doit rester sur l'appareil.
- **Pièges à éviter** : comparer des perplexités avec des vocabulaires différents, laisser des mots de test entrer dans le vocabulaire (fuite), oublier `</s>` dans les probabilités.

---

<a id="extensions"></a>
## <img src="assets/icons/compass.svg" width="26" height="26" align="top">&nbsp;11. Pistes d'extension

- Lissage de **Kneser-Ney**.
- Entraînement sur **son propre texte** (un export WhatsApp est une donnée personnelle : uniquement avec son consentement et sans jamais le partager).
- Jeu de **validation séparé** pour régler k et les λ, puis un unique passage sur le test.
- Optimisation des poids d'interpolation par validation.
- Passage vers un modèle neuronal (voir le Lab 07).

---

<p align="center"><sub>Lab 02 · Smart Keyboard V1 · n-grammes + add-k + back-off + interpolation</sub></p>