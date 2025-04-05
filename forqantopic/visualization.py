import fireducks.pandas as pd
import numpy as np

import plotly.express as px

def plot_umap_scatter(reduced_embeddings, labels, texts=None, topic_names=None):
    import fireducks.pandas as pd
    import numpy as np
    import plotly.express as px

    labels = np.array(labels)
    dims = reduced_embeddings.shape[1]

    if dims < 2:
        raise ValueError("Reduced embeddings must have at least 2 dimensions for plotting.")

    # Decide how many dims to visualize
    if dims == 2:
        axis_labels = ["X", "Y"]
        used_dims = 2
    else:
        axis_labels = ["X", "Y", "Z"]
        used_dims = 3

    df = pd.DataFrame(reduced_embeddings[:, :used_dims], columns=axis_labels)
    df["cluster"] = labels
    df["text"] = texts if texts else [""] * len(labels)

    if topic_names:
        df["topic"] = [topic_names.get(label, "Noise") if label != -1 else "Noise" for label in labels]
    else:
        df["topic"] = df["cluster"].astype(str)

    if used_dims == 2:
        fig = px.scatter(
            df,
            x="X",
            y="Y",
            color="topic",
            hover_data=["text"],
            title="UMAP 2D Projection of Text Clusters"
        )
    else:
        fig = px.scatter_3d(
            df,
            x="X",
            y="Y",
            z="Z",
            color="topic",
            hover_data=["text"],
            title="UMAP 3D Projection of Text Clusters"
        )

    fig.show()
