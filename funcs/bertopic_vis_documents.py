import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bertopic._utils import check_documents_type, validate_distance_matrix
from bertopic.plotting._hierarchy import _get_annotations
import plotly.figure_factory as ff
from packaging import version

import math
from umap import UMAP
from typing import List, Union, Callable

from scipy.sparse import csr_matrix
from scipy.cluster import hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import __version__ as sklearn_version
from tqdm import tqdm

import itertools
import numpy as np

# Shamelessly taken and adapted from Bertopic original implementation here (Maarten Grootendorst): https://github.com/MaartenGr/BERTopic/blob/master/bertopic/plotting/_documents.py

def visualize_documents_custom(topic_model,
                        docs: List[str],
                        hover_labels: List[str],
                        topics: List[int] = None,
                        embeddings: np.ndarray = None,
                        reduced_embeddings: np.ndarray = None,
                        sample: float = None,
                        hide_annotations: bool = False,
                        hide_document_hover: bool = False,
                        custom_labels: Union[bool, str] = False,
                        title: str = "<b>Documents and Topics</b>",
                        width: int = 1200,
                        height: int = 750, progress=gr.Progress(track_tqdm=True)):
    """ Visualize documents and their topics in 2D

    Arguments:
        topic_model: A fitted BERTopic instance.
        docs: The documents you used when calling either `fit` or `fit_transform`
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`.
        embeddings: The embeddings of all documents in `docs`.
        reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
        sample: The percentage of documents in each topic that you would like to keep.
                Value can be between 0 and 1. Setting this value to, for example,
                0.1 (10% of documents in each topic) makes it easier to visualize
                millions of documents as a subset is chosen.
        hide_annotations: Hide the names of the traces on top of each cluster.
        hide_document_hover: Hide the content of the documents when hovering over
                             specific points. Helps to speed up generation of visualization.
        custom_labels: If bool, whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:

    To visualize the topics simply run:

    ```python
    topic_model.visualize_documents(docs)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and prefered pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Prepare embeddings
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic
    topic_model = BERTopic().fit(docs, embeddings)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_documents(docs, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)
    fig.write_html("path/to/file.html")
    ```

    <iframe src="../../getting_started/visualization/documents.html"
    style="width:1000px; height: 800px; border: 0px;""></iframe>
    """
    topic_per_doc = topic_model.topics_

    # Add <br> tags to hover labels to get them to appear on multiple lines
    def wrap_by_word(s, n):
        '''returns a string up to 300 words where \\n is inserted between every n words'''
        a = s.split()[:300]
        ret = ''
        for i in range(0, len(a), n):
            ret += ' '.join(a[i:i+n]) + '<br>'
        return ret
    
    # Apply the function to every element in the list
    hover_labels = [wrap_by_word(s, n=20) for s in hover_labels]


    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["hover_labels"] = [hover_labels[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit(embeddings_to_reduce)
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    unique_topics = set(topic_per_doc)
    if topics is None:
        topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Prepare text and names
    if isinstance(custom_labels, str):
        names = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in unique_topics]
        names = ["_".join([label[0] for label in labels[:4]]) for labels in names]
        names = [label if len(label) < 30 else label[:27] + "..." for label in names]
    elif topic_model.custom_labels_ is not None and custom_labels:
        print("Using custom labels: ", topic_model.custom_labels_)
        names = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in unique_topics]
    else:
        print("Not using custom labels")
        names = [f"{topic} " + ", ".join([word for word, value in topic_model.get_topic(topic)][:3]) for topic in unique_topics]

    #print(names)

    # Visualize
    fig = go.Figure()

    # Outliers and non-selected topics
    non_selected_topics = set(unique_topics).difference(topics)
    if len(non_selected_topics) == 0:
        non_selected_topics = [-1]

    selection = df.loc[df.topic.isin(non_selected_topics), :]
    selection["text"] = ""
    selection.loc[len(selection), :] = [None, None, None, selection.x.mean(), selection.y.mean(), "Other documents"]

    fig.add_trace(
        go.Scattergl(
            x=selection.x,
            y=selection.y,
            hovertext=selection.hover_labels if not hide_document_hover else None,
            hoverinfo="text",
            mode='markers+text',
            name="other",
            showlegend=False,
            marker=dict(color='#CFD8DC', size=5, opacity=0.5),
            hoverlabel=dict(align='left')
        )
    )

    # Selected topics
    for name, topic in zip(names, unique_topics):
        #print(name)
        #print(topic)
        if topic in topics and topic != -1:
            selection = df.loc[df.topic == topic, :]
            selection["text"] = ""

            if not hide_annotations:
                selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

            fig.add_trace(
                go.Scattergl(
                    x=selection.x,
                    y=selection.y,
                    hovertext=selection.hover_labels if not hide_document_hover else None,
                    hoverinfo="text",
                    text=selection.text,
                    mode='markers+text',
                    name=name,
                    textfont=dict(
                        size=12,
                    ),
                    marker=dict(size=5, opacity=0.5),
                    hoverlabel=dict(align='left')
            ))

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        hoverlabel_align = 'left',
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def hierarchical_topics_custom(self,
                        docs: List[str],
                        linkage_function: Callable[[csr_matrix], np.ndarray] = None,
                        distance_function: Callable[[csr_matrix], csr_matrix] = None, progress=gr.Progress(track_tqdm=True)) -> pd.DataFrame:
    """ Create a hierarchy of topics

    To create this hierarchy, BERTopic needs to be already fitted once.
    Then, a hierarchy is calculated on the distance matrix of the c-TF-IDF
    representation using `scipy.cluster.hierarchy.linkage`.

    Based on that hierarchy, we calculate the topic representation at each
    merged step. This is a local representation, as we only assume that the
    chosen step is merged and not all others which typically improves the
    topic representation.

    Arguments:
        docs: The documents you used when calling either `fit` or `fit_transform`
        linkage_function: The linkage function to use. Default is:
                            `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
        distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                            `lambda x: 1 - cosine_similarity(x)`.
                            You can pass any function that returns either a square matrix of 
                            shape (n_samples, n_samples) with zeros on the diagonal and 
                            non-negative values or condensed distance matrix of shape
                            (n_samples * (n_samples - 1) / 2,) containing the upper
                            triangular of the distance matrix.

    Returns:
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                                represented by their parents and their children

    Examples:

    ```python
    from bertopic import BERTopic
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    ```

    A custom linkage function can be used as follows:

    ```python
    from scipy.cluster import hierarchy as sch
    from bertopic import BERTopic
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs)

    # Hierarchical topics
    linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)
    hierarchical_topics = topic_model.hierarchical_topics(docs, linkage_function=linkage_function)
    ```
    """
    check_documents_type(docs)
    if distance_function is None:
        distance_function = lambda x: 1 - cosine_similarity(x)

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

    # Calculate distance
    embeddings = self.c_tf_idf_[self._outliers:]
    X = distance_function(embeddings)
    X = validate_distance_matrix(X, embeddings.shape[0])

    # Use the 1-D condensed distance matrix as an input instead of the raw distance matrix
    Z = linkage_function(X)

    # Calculate basic bag-of-words to be iteratively merged later
    documents = pd.DataFrame({"Document": docs,
                                "ID": range(len(docs)),
                                "Topic": self.topics_})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    documents_per_topic = documents_per_topic.loc[documents_per_topic.Topic != -1, :]
    clean_documents = self._preprocess_text(documents_per_topic.Document.values)

    # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
    # and will be removed in 1.2. Please use get_feature_names_out instead.
    if version.parse(sklearn_version) >= version.parse("1.0.0"):
        words = self.vectorizer_model.get_feature_names_out()
    else:
        words = self.vectorizer_model.get_feature_names()

    bow = self.vectorizer_model.transform(clean_documents)

    # Extract clusters
    hier_topics = pd.DataFrame(columns=["Parent_ID", "Parent_Name", "Topics",
                                        "Child_Left_ID", "Child_Left_Name",
                                        "Child_Right_ID", "Child_Right_Name"])
    for index in tqdm(range(len(Z))):

        # Find clustered documents
        clusters = sch.fcluster(Z, t=Z[index][2], criterion='distance') - self._outliers
        nr_clusters = len(clusters)

        # Extract first topic we find to get the set of topics in a merged topic
        topic = None
        val = Z[index][0]
        while topic is None:
            if val - len(clusters) < 0:
                topic = int(val)
            else:
                val = Z[int(val - len(clusters))][0]
        clustered_topics = [i for i, x in enumerate(clusters) if x == clusters[topic]]

        # Group bow per cluster, calculate c-TF-IDF and extract words
        grouped = csr_matrix(bow[clustered_topics].sum(axis=0))
        c_tf_idf = self.ctfidf_model.transform(grouped)
        selection = documents.loc[documents.Topic.isin(clustered_topics), :]
        selection.Topic = 0
        words_per_topic = self._extract_words_per_topic(words, selection, c_tf_idf, calculate_aspects=False)

        # Extract parent's name and ID
        parent_id = index + len(clusters)
        parent_name = ", ".join([x[0] for x in words_per_topic[0]][:5])

        # Extract child's name and ID
        Z_id = Z[index][0]
        child_left_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

        if Z_id - nr_clusters < 0:
            child_left_name = ", ".join([x[0] for x in self.get_topic(Z_id)][:5])
        else:
            child_left_name = hier_topics.iloc[int(child_left_id)].Parent_Name

        # Extract child's name and ID
        Z_id = Z[index][1]
        child_right_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

        if Z_id - nr_clusters < 0:
            child_right_name = ", ".join([x[0] for x in self.get_topic(Z_id)][:5])
        else:
            child_right_name = hier_topics.iloc[int(child_right_id)].Parent_Name

        # Save results
        hier_topics.loc[len(hier_topics), :] = [parent_id, parent_name,
                                                clustered_topics,
                                                int(Z[index][0]), child_left_name,
                                                int(Z[index][1]), child_right_name]

    hier_topics["Distance"] = Z[:, 2]
    hier_topics = hier_topics.sort_values("Parent_ID", ascending=False)
    hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]] = hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]].astype(str)

    return hier_topics

def visualize_hierarchy_custom(topic_model,
                        orientation: str = "left",
                        topics: List[int] = None,
                        top_n_topics: int = None,
                        custom_labels: Union[bool, str] = False,
                        title: str = "<b>Hierarchical Clustering</b>",
                        width: int = 1000,
                        height: int = 600,
                        hierarchical_topics: pd.DataFrame = None,
                        linkage_function: Callable[[csr_matrix], np.ndarray] = None,
                        distance_function: Callable[[csr_matrix], csr_matrix] = None,
                        color_threshold: int = 1) -> go.Figure:
    """ Visualize a hierarchical structure of the topics

    A ward linkage function is used to perform the
    hierarchical clustering based on the cosine distance
    matrix between topic embeddings.

    Arguments:
        topic_model: A fitted BERTopic instance.
        orientation: The orientation of the figure.
                     Either 'left' or 'bottom'
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        custom_labels: If bool, whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
                       NOTE: Custom labels are only generated for the original 
                       un-merged topics.
        title: Title of the plot.
        width: The width of the figure. Only works if orientation is set to 'left'
        height: The height of the figure. Only works if orientation is set to 'bottom'
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                             represented by their parents and their children.
                             NOTE: The hierarchical topic names are only visualized
                             if both `topics` and `top_n_topics` are not set.
        linkage_function: The linkage function to use. Default is:
                          `lambda x: sch.linkage(x, 'ward', optimal_ordering=True)`
                          NOTE: Make sure to use the same `linkage_function` as used
                          in `topic_model.hierarchical_topics`.
        distance_function: The distance function to use on the c-TF-IDF matrix. Default is:
                           `lambda x: 1 - cosine_similarity(x)`.
                            You can pass any function that returns either a square matrix of 
                            shape (n_samples, n_samples) with zeros on the diagonal and 
                            non-negative values or condensed distance matrix of shape 
                            (n_samples * (n_samples - 1) / 2,) containing the upper 
                            triangular of the distance matrix.
                           NOTE: Make sure to use the same `distance_function` as used
                           in `topic_model.hierarchical_topics`.
        color_threshold: Value at which the separation of clusters will be made which
                         will result in different colors for different clusters.
                         A higher value will typically lead in less colored clusters.

    Returns:
        fig: A plotly figure

    Examples:

    To visualize the hierarchical structure of
    topics simply run:

    ```python
    topic_model.visualize_hierarchy()
    ```

    If you also want the labels visualized of hierarchical topics,
    run the following:

    ```python
    # Extract hierarchical topics and their representations
    hierarchical_topics = topic_model.hierarchical_topics(docs)

    # Visualize these representations
    topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    ```

    If you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_hierarchy()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/hierarchy.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    if distance_function is None:
        distance_function = lambda x: 1 - cosine_similarity(x)

    if linkage_function is None:
        linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Select embeddings
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])

    # Select topic embeddings
    if topic_model.c_tf_idf_ is not None:
        embeddings = topic_model.c_tf_idf_[indices]
    else:
        embeddings = np.array(topic_model.topic_embeddings_)[indices]
        
    # Annotations
    if hierarchical_topics is not None and len(topics) == len(freq_df.Topic.to_list()):
        annotations = _get_annotations(topic_model=topic_model,
                                       hierarchical_topics=hierarchical_topics,
                                       embeddings=embeddings,
                                       distance_function=distance_function,
                                       linkage_function=linkage_function,
                                       orientation=orientation,
                                       custom_labels=custom_labels)
    else:
        annotations = None

    # wrap distance function to validate input and return a condensed distance matrix
    distance_function_viz = lambda x: validate_distance_matrix(
        distance_function(x), embeddings.shape[0])
    # Create dendogram
    fig = ff.create_dendrogram(embeddings,
                               orientation=orientation,
                               distfun=distance_function_viz,
                               linkagefun=linkage_function,
                               hovertext=annotations,
                               color_threshold=color_threshold)

    # Create nicer labels
    axis = "yaxis" if orientation == "left" else "xaxis"
    if isinstance(custom_labels, str):
        new_labels = [[[str(x), None]] + topic_model.topic_aspects_[custom_labels][x] for x in fig.layout[axis]["ticktext"]]
        new_labels = [", ".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]
    elif topic_model.custom_labels_ is not None and custom_labels:
        new_labels = [topic_model.custom_labels_[topics[int(x)] + topic_model._outliers] for x in fig.layout[axis]["ticktext"]]
    else:
        new_labels = [[[str(topics[int(x)]), None]] + topic_model.get_topic(topics[int(x)])
                      for x in fig.layout[axis]["ticktext"]]
        new_labels = [", ".join([label[0] for label in labels[:4]]) for labels in new_labels]
        new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]

    # Stylize layout
    fig.update_layout(
        plot_bgcolor='#ECEFF1',
        template="plotly_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    # Stylize orientation
    if orientation == "left":
        fig.update_layout(height=200 + (15 * len(topics)),
                          width=width,
                          yaxis=dict(tickmode="array",
                                     ticktext=new_labels))

        # Fix empty space on the bottom of the graph
        y_max = max([trace['y'].max() + 5 for trace in fig['data']])
        y_min = min([trace['y'].min() - 5 for trace in fig['data']])
        fig.update_layout(yaxis=dict(range=[y_min, y_max]))

    else:
        fig.update_layout(width=200 + (15 * len(topics)),
                          height=height,
                          xaxis=dict(tickmode="array",
                                     ticktext=new_labels))

    if hierarchical_topics is not None:
        for index in [0, 3]:
            axis = "x" if orientation == "left" else "y"
            xs = [data["x"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            ys = [data["y"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]
            hovertext = [data["text"][index] for data in fig.data if (data["text"] and data[axis][index] > 0)]

            fig.add_trace(go.Scatter(x=xs, y=ys, marker_color='black',
                                     hovertext=hovertext, hoverinfo="text",
                                     mode='markers', showlegend=False))
    return fig

def visualize_hierarchical_documents_custom(topic_model,
                                     docs: List[str],
                                     hover_labels: List[str],
                                     hierarchical_topics: pd.DataFrame,
                                     topics: List[int] = None,
                                     embeddings: np.ndarray = None,
                                     reduced_embeddings: np.ndarray = None,
                                     sample: Union[float, int] = None,
                                     hide_annotations: bool = False,
                                     hide_document_hover: bool = True,
                                     nr_levels: int = 10,
                                     level_scale: str = 'linear', 
                                     custom_labels: Union[bool, str] = False,
                                     title: str = "<b>Hierarchical Documents and Topics</b>",
                                     width: int = 1200,
                                     height: int = 750, progress=gr.Progress(track_tqdm=True)) -> go.Figure:
    """ Visualize documents and their topics in 2D at different levels of hierarchy

    Arguments:
        docs: The documents you used when calling either `fit` or `fit_transform`
        hierarchical_topics: A dataframe that contains a hierarchy of topics
                             represented by their parents and their children
        topics: A selection of topics to visualize.
                Not to be confused with the topics that you get from `.fit_transform`.
                For example, if you want to visualize only topics 1 through 5:
                `topics = [1, 2, 3, 4, 5]`.
        embeddings: The embeddings of all documents in `docs`.
        reduced_embeddings: The 2D reduced embeddings of all documents in `docs`.
        sample: The percentage of documents in each topic that you would like to keep.
                Value can be between 0 and 1. Setting this value to, for example,
                0.1 (10% of documents in each topic) makes it easier to visualize
                millions of documents as a subset is chosen.
        hide_annotations: Hide the names of the traces on top of each cluster.
        hide_document_hover: Hide the content of the documents when hovering over
                             specific points. Helps to speed up generation of visualizations.
        nr_levels: The number of levels to be visualized in the hierarchy. First, the distances
                   in `hierarchical_topics.Distance` are split in `nr_levels` lists of distances. 
                   Then, for each list of distances, the merged topics are selected that have a 
                   distance less or equal to the maximum distance of the selected list of distances.
                   NOTE: To get all possible merged steps, make sure that `nr_levels` is equal to
                   the length of `hierarchical_topics`.
        level_scale: Whether to apply a linear or logarithmic (log) scale levels of the distance 
                     vector. Linear scaling will perform an equal number of merges at each level 
                     while logarithmic scaling will perform more mergers in earlier levels to 
                     provide more resolution at higher levels (this can be used for when the number 
                     of topics is large). 
        custom_labels: If bool, whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
                       NOTE: Custom labels are only generated for the original 
                       un-merged topics.
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.

    Examples:

    To visualize the topics simply run:

    ```python
    topic_model.visualize_hierarchical_documents(docs, hierarchical_topics)
    ```

    Do note that this re-calculates the embeddings and reduces them to 2D.
    The advised and prefered pipeline for using this function is as follows:

    ```python
    from sklearn.datasets import fetch_20newsgroups
    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from umap import UMAP

    # Prepare embeddings
    docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)

    # Train BERTopic and extract hierarchical topics
    topic_model = BERTopic().fit(docs, embeddings)
    hierarchical_topics = topic_model.hierarchical_topics(docs)

    # Reduce dimensionality of embeddings, this step is optional
    # reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # Run the visualization with the original embeddings
    topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, embeddings=embeddings)

    # Or, if you have reduced the original embeddings already:
    topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings)
    fig.write_html("path/to/file.html")
    ```

    NOTE:
        This visualization was inspired by the scatter plot representation of Doc2Map:
        https://github.com/louisgeisler/Doc2Map

    <iframe src="../../getting_started/visualization/hierarchical_documents.html"
    style="width:1000px; height: 770px; border: 0px;""></iframe>
    """
    topic_per_doc = topic_model.topics_

    # Add <br> tags to hover labels to get them to appear on multiple lines
    def wrap_by_word(s, n):
        '''returns a string up to 300 words where \\n is inserted between every n words'''
        a = s.split()[:300]
        ret = ''
        for i in range(0, len(a), n):
            ret += ' '.join(a[i:i+n]) + '<br>'
        return ret
    
    # Apply the function to every element in the list
    hover_labels = [wrap_by_word(s, n=20) for s in hover_labels]

    # Sample the data to optimize for visualization and dimensionality reduction
    if sample is None or sample > 1:
        sample = 1

    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s)*sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["hover_labels"] = [hover_labels[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]

    # Extract embeddings if not already done
    if sample is None:
        if embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")
        else:
            embeddings_to_reduce = embeddings
    else:
        if embeddings is not None:
            embeddings_to_reduce = embeddings[indices]
        elif embeddings is None and reduced_embeddings is None:
            embeddings_to_reduce = topic_model._extract_embeddings(df.doc.to_list(), method="document")

    # Reduce input embeddings
    if reduced_embeddings is None:
        umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit(embeddings_to_reduce)
        embeddings_2d = umap_model.embedding_
    elif sample is not None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings[indices]
    elif sample is None and reduced_embeddings is not None:
        embeddings_2d = reduced_embeddings

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    # Create topic list for each level, levels are created by calculating the distance
    distances = hierarchical_topics.Distance.to_list()
    if level_scale == 'log' or level_scale == 'logarithmic':
        log_indices = np.round(np.logspace(start=math.log(1,10), stop=math.log(len(distances)-1,10), num=nr_levels)).astype(int).tolist()
        log_indices.reverse()
        max_distances = [distances[i] for i in log_indices]
    elif level_scale == 'lin' or level_scale == 'linear':
        max_distances = [distances[indices[-1]] for indices in np.array_split(range(len(hierarchical_topics)), nr_levels)][::-1]
    else:
        raise ValueError("level_scale needs to be one of 'log' or 'linear'")
    
    for index, max_distance in enumerate(max_distances):

        # Get topics below `max_distance`
        mapping = {topic: topic for topic in df.topic.unique()}
        selection = hierarchical_topics.loc[hierarchical_topics.Distance <= max_distance, :]
        selection.Parent_ID = selection.Parent_ID.astype(int)
        selection = selection.sort_values("Parent_ID")

        for row in selection.iterrows():
            for topic in row[1].Topics:
                mapping[topic] = row[1].Parent_ID

        # Make sure the mappings are mapped 1:1
        mappings = [True for _ in mapping]
        while any(mappings):
            for i, (key, value) in enumerate(mapping.items()):
                if value in mapping.keys() and key != value:
                    mapping[key] = mapping[value]
                else:
                    mappings[i] = False

        # Create new column
        df[f"level_{index+1}"] = df.topic.map(mapping)
        df[f"level_{index+1}"] = df[f"level_{index+1}"].astype(int)

    # Prepare topic names of original and merged topics
    trace_names = []
    topic_names = {}
    trace_name_char_length = 60
    for topic in range(hierarchical_topics.Parent_ID.astype(int).max()):
        if topic < hierarchical_topics.Parent_ID.astype(int).min():
            if topic_model.get_topic(topic):
                if isinstance(custom_labels, str):
                    trace_name = f"{topic} " + ", ".join(list(zip(*topic_model.topic_aspects_[custom_labels][topic]))[0][:5])
                elif topic_model.custom_labels_ is not None and custom_labels:
                    trace_name = topic_model.custom_labels_[topic + topic_model._outliers]
                else:
                    trace_name = f"{topic} " + ", ".join([word[:20] for word, _ in topic_model.get_topic(topic)][:5])
                topic_names[topic] = {"trace_name": trace_name[:trace_name_char_length], "plot_text": trace_name[:trace_name_char_length]}
                trace_names.append(trace_name)
        else:
            trace_name = f"{topic} " + hierarchical_topics.loc[hierarchical_topics.Parent_ID == str(topic), "Parent_Name"].values[0]
            plot_text = ", ".join([name[:20] for name in trace_name.split(" ")[:5]])
            topic_names[topic] = {"trace_name": trace_name[:trace_name_char_length], "plot_text": plot_text[:trace_name_char_length]}
            trace_names.append(trace_name)

    # Prepare traces
    all_traces = []
    for level in range(len(max_distances)):
        traces = []

        # Outliers
        if topic_model._outliers:
            traces.append(
                    go.Scattergl(
                        x=df.loc[(df[f"level_{level+1}"] == -1), "x"],
                        y=df.loc[df[f"level_{level+1}"] == -1, "y"],
                        mode='markers+text',
                        name="other",
                        hoverinfo="text",
                        hovertext=df.loc[(df[f"level_{level+1}"] == -1), "hover_labels"] if not hide_document_hover else None,
                        showlegend=False,
                        marker=dict(color='#CFD8DC', size=5, opacity=0.5),
                        hoverlabel=dict(align='left')
                    )
                )

        # Selected topics
        if topics:
            selection = df.loc[(df.topic.isin(topics)), :]
            unique_topics = sorted([int(topic) for topic in selection[f"level_{level+1}"].unique()])
        else:
            unique_topics = sorted([int(topic) for topic in df[f"level_{level+1}"].unique()])

        for topic in unique_topics:
            if topic != -1:
                if topics:
                    selection = df.loc[(df[f"level_{level+1}"] == topic) &
                                       (df.topic.isin(topics)), :]
                else:
                    selection = df.loc[df[f"level_{level+1}"] == topic, :]

                if not hide_annotations:
                    selection.loc[len(selection), :] = None
                    selection["text"] = ""
                    selection.loc[len(selection) - 1, "x"] = selection.x.mean()
                    selection.loc[len(selection) - 1, "y"] = selection.y.mean()
                    selection.loc[len(selection) - 1, "text"] = topic_names[int(topic)]["plot_text"]

                traces.append(
                    go.Scattergl(
                        x=selection.x,
                        y=selection.y,
                        text=selection.text if not hide_annotations else None,
                        hovertext=selection.hover_labels if not hide_document_hover else None,
                        hoverinfo="text",
                        name=topic_names[int(topic)]["trace_name"],
                        mode='markers+text',
                        marker=dict(size=5, opacity=0.5),
                        hoverlabel=dict(align='left')
                    )
                )

        all_traces.append(traces)

    # Track and count traces
    nr_traces_per_set = [len(traces) for traces in all_traces]
    trace_indices = [(0, nr_traces_per_set[0])]
    for index, nr_traces in enumerate(nr_traces_per_set[1:]):
        start = trace_indices[index][1]
        end = nr_traces + start
        trace_indices.append((start, end))

    # Visualization
    fig = go.Figure()
    for traces in all_traces:
        for trace in traces:
            fig.add_trace(trace)

    for index in range(len(fig.data)):
        if index >= nr_traces_per_set[0]:
            fig.data[index].visible = False

    # Create and add slider
    steps = []
    for index, indices in enumerate(trace_indices):
        step = dict(
            method="update",
            label=str(index),
            args=[{"visible": [False] * len(fig.data)}]
        )
        for index in range(indices[1]-indices[0]):
            step["args"][0]["visible"][index+indices[0]] = True
        steps.append(step)

    sliders = [dict(
        currentvalue={"prefix": "Level: "},
        pad={"t": 20},
        steps=steps
    )]

    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        sliders=sliders,
        template="simple_white",
        title={
            'text': f"{title}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    hierarchy_topics_df = df.filter(regex=r'topic|^level').drop_duplicates(subset="topic")

    topic_names = pd.DataFrame(topic_names).T


    return fig, hierarchy_topics_df, topic_names

def visualize_barchart_custom(topic_model,
                       topics: List[int] = None,
                       top_n_topics: int = 8,
                       n_words: int = 5,
                       custom_labels: Union[bool, str] = False,
                       title: str = "<b>Topic Word Scores</b>",
                       width: int = 250,
                       height: int = 250, progress=gr.Progress(track_tqdm=True)) -> go.Figure:
    """ Visualize a barchart of selected topics

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        custom_labels: If bool, whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of each figure.
        height: The height of each figure.

    Returns:
        fig: A plotly figure

    Examples:

    To visualize the barchart of selected topics
    simply run:

    ```python
    topic_model.visualize_barchart()
    ```

    Or if you want to save the resulting figure:

    ```python
    fig = topic_model.visualize_barchart()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/bar_chart.html"
    style="width:1100px; height: 660px; border: 0px;""></iframe>
    """
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:6])

    # Initialize figure
    if isinstance(custom_labels, str):
        subplot_titles = [[[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic] for topic in topics]
        subplot_titles = ["_".join([label[0] for label in labels[:4]]) for labels in subplot_titles]
        subplot_titles = [label if len(label) < 30 else label[:27] + "..." for label in subplot_titles]
    elif topic_model.custom_labels_ is not None and custom_labels:
        subplot_titles = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topics]
    else:
        subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = 3
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{title}",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=14,
                color="Black")
        },
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig