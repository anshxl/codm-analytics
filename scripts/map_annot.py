import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import argparse

def annotate_minimap_with_routes(image_path, map, setup, save_csv="annotations.csv", save_img="annotated.png"):
    """
    Manual annotation tool: click points on minimap image to mark FK locations or routes.
    Draws lines/arrows between sequential clicks.
    Saves both coordinates and an annotated image.

    Args:
        image_path (str): Path to minimap image.
        n_points (int): Max number of points to click before finishing (close window when done).
        save_csv (str): File to save clicked coordinates.
        save_img (str): File to save annotated minimap.
    """
    # Load minimap
    img = mpimg.imread(image_path)

    # Show image for clicks
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Click up to 1000 points (close window when done)")

    try:
        # Collect clicks
        coords = plt.ginput(1000, timeout=0)  # unlimited time
    except Exception as e:
        print(f"Annotation interrupted: {e}")
        coords = []
    finally:
        plt.close(fig)

    if not coords:
        print("No points clicked.")
        return None

    # Save annotated image with dots + arrows
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    for (x, y) in coords:
        ax.plot(x, y, "ro", markersize=2)  # red dot
        # ax.text(x + 5, y - 5, str(i + 1), color="yellow", fontsize=10)
        # if i > 0:
        #     # draw arrow from previous point
        #     prev_x, prev_y = coords[i - 1]
        #     ax.annotate("",
        #                 xy=(x, y), xytext=(prev_x, prev_y),
        #                 arrowprops=dict(arrowstyle="->", color="cyan", lw=2))
    ax.set_title(f"{setup} on {map}")
    fig.savefig(save_img)
    plt.close(fig)

    # Save coords to CSV
    df = pd.DataFrame(coords, columns=["x", "y"])
    # Check if file exists to append or create new
    try:
        existing_df = pd.read_csv(save_csv)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(save_csv, index=False)
    return df

def main():
    parser = argparse.ArgumentParser(description="Annotate minimap with FK locations/routes.")
    parser.add_argument("Map", type=str, help="Map of interest")
    parser.add_argument("--setup", type=str, default="3A_Stack", help="Setup Archetype")
    args = parser.parse_args()
    args.image_path = f"Minimaps/{args.Map}/map.jpeg"
    save_path = f"Minimaps/{args.Map}/{args.setup}"
    args.save_csv = f"{save_path}_annotations.csv"
    args.save_img = f"{save_path}_annotated.png"
    annotate_minimap_with_routes(args.image_path, args.save_csv, args.save_img)

if __name__ == "__main__":
    main()