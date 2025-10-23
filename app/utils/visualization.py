"""
Visualization Utilities
Embedding space visualization, attention maps, result display
"""

from typing import List, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_embeddings_pca(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Embedding Space (PCA)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize embeddings using PCA.
    
    Args:
        embeddings: Embedding vectors (N, D)
        labels: Optional labels for coloring
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, ax=ax, label='User ID')
    else:
        ax.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            alpha=0.6,
            s=50
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_embeddings_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    perplexity: int = 30,
    title: str = "Embedding Space (t-SNE)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Embedding vectors (N, D)
        labels: Optional labels for coloring
        perplexity: t-SNE perplexity parameter
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, ax=ax, label='User ID')
    else:
        ax.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            alpha=0.6,
            s=50
        )
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_similarity_matrix(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Cosine Similarity Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize pairwise similarity matrix.
    
    Args:
        embeddings: Embedding vectors (N, D)
        labels: Optional labels for axes
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Compute similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=labels if labels else False,
        yticklabels=labels if labels else False,
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def draw_detection_result(
    image: np.ndarray,
    bbox: np.ndarray,
    landmarks: dict,
    confidence: float,
    label: str = "Face",
    color: tuple = (0, 255, 0)
) -> np.ndarray:
    """
    Draw detection result on image.
    
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        landmarks: Facial landmarks
        confidence: Detection confidence
        label: Label text
        color: Box color (BGR)
        
    Returns:
        np.ndarray: Image with drawn results
    """
    result_img = image.copy()
    
    # Draw bounding box
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
    
    # Draw landmarks
    for name, (x, y) in landmarks.items():
        cv2.circle(result_img, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # Draw label
    label_text = f"{label}: {confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        result_img, 
        (x1, y1 - text_height - 10), 
        (x1 + text_width, y1), 
        color, 
        -1
    )
    cv2.putText(
        result_img, 
        label_text, 
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (255, 255, 255), 
        2
    )
    
    return result_img


def draw_authentication_result(
    image: np.ndarray,
    bbox: np.ndarray,
    authenticated: bool,
    confidence: float,
    user_id: Optional[str] = None,
    liveness_score: Optional[float] = None
) -> np.ndarray:
    """
    Draw authentication result on image.
    
    Args:
        image: Input image
        bbox: Face bounding box
        authenticated: Authentication result
        confidence: Verification confidence
        user_id: User identifier
        liveness_score: Liveness score
        
    Returns:
        np.ndarray: Image with drawn results
    """
    result_img = image.copy()
    
    # Color based on result
    color = (0, 255, 0) if authenticated else (0, 0, 255)
    
    # Draw bounding box
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
    
    # Prepare text
    if authenticated:
        if user_id:
            status_text = f"Authenticated: {user_id}"
        else:
            status_text = "Authenticated"
    else:
        status_text = "Not Authenticated"
    
    # Draw status
    y_offset = y1 - 10
    cv2.putText(
        result_img, 
        status_text,
        (x1, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        color, 
        2
    )
    
    # Draw confidence
    y_offset -= 25
    cv2.putText(
        result_img, 
        f"Confidence: {confidence:.2f}",
        (x1, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        color, 
        2
    )
    
    # Draw liveness score if available
    if liveness_score is not None:
        y_offset -= 25
        liveness_color = (0, 255, 0) if liveness_score > 0.9 else (0, 165, 255)
        cv2.putText(
            result_img, 
            f"Liveness: {liveness_score:.2f}",
            (x1, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            liveness_color, 
            2
        )
    
    return result_img


def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str],
    rows: int = 2,
    cols: int = 3,
    figsize: tuple = (15, 10)
) -> plt.Figure:
    """
    Create a grid comparison of images.
    
    Args:
        images: List of images
        titles: List of titles
        rows: Number of rows
        cols: Number of columns
        figsize: Figure size
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, (image, title) in enumerate(zip(images, titles)):
        if idx >= len(axes):
            break
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        axes[idx].imshow(image_rgb)
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    return fig


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image (for attention/Grad-CAM visualization).
    
    Args:
        image: Original image
        heatmap: Heatmap (0-1 normalized)
        alpha: Overlay transparency
        colormap: OpenCV colormap
        
    Returns:
        np.ndarray: Image with overlaid heatmap
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to color
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        colormap
    )
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def plot_model_comparison(
    model_names: List[str],
    accuracies: List[float],
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart comparing model performances.
    
    Args:
        model_names: List of model names
        accuracies: List of accuracy scores
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(model_names, accuracies, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., 
            height,
            f'{height:.3f}',
            ha='center', 
            va='bottom'
        )
    
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_latency_comparison(
    operations: List[str],
    latencies: List[float],
    title: str = "Latency Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot latency comparison for different operations.
    
    Args:
        operations: List of operation names
        latencies: List of latency values (ms)
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(operations, latencies, color='coral', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width, 
            bar.get_y() + bar.get_height() / 2.,
            f'{width:.1f} ms',
            ha='left', 
            va='center',
            fontsize=10
        )
    
    ax.set_xlabel('Latency (ms)')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

