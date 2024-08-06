from manim import *
import numpy as np


class KernelAnimation(Scene):
    def construct(self):

        # Create the image
        img = (
            ImageMobject("./images/mnist_7.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(20)
        )
        # Create the grid overlay
        grid = NumberPlane(
            x_range=[-14, 14, 1],
            y_range=[-14, 14, 1],
            background_line_style={
                "stroke_color": WHITE,
                "stroke_width": 1,
                "stroke_opacity": 0.6,
            },
            axis_config={
                "stroke_color": WHITE,
                "stroke_width": 1,
                "stroke_opacity": 0.6,
                "include_numbers": False,
            },
            faded_line_ratio=0,
        )

        cell_width = img.width / 28

        # Scale it to the size of the image
        grid.scale_to_fit_width(img.width).move_to(img)
        img_rect = SurroundingRectangle(img, color=GRAY, stroke_width=1, buff=0)
        img = Group(img, grid, img_rect)

        # Send image to edge
        img.to_edge(RIGHT, buff=1)

        # Create the filter
        filter = NumberPlane(
            x_range=[-2, 1, 1],
            y_range=[-2, 1, 1],
            background_line_style={
                "stroke_color": BLUE,
                "stroke_width": 1,
                "stroke_opacity": 1,
            },
            axis_config={
                "stroke_color": BLUE,
                "stroke_width": 1,
                "stroke_opacity": 1,
                "include_numbers": False,
            },
            faded_line_ratio=0,
        )

        filter.add(SurroundingRectangle(filter, color=BLUE, stroke_width=1, buff=0))

        # Scale it to 3 times the size of a cell in the image
        filter.scale_to_fit_width(3 * cell_width).to_edge(LEFT, buff=1)

        self.play(FadeIn(img))
        self.play(FadeIn(filter))

        # Move the filter to the center of the image
        # Remember to offset it by half the width of a cell
        self.play(
            filter.animate.move_to(img.get_center() + 0.5 * cell_width * (RIGHT + DOWN))
        )

        self.wait(2)

        # Move the filter around
        self.play(filter.animate.shift(cell_width * RIGHT))
        self.play(filter.animate.shift(cell_width * RIGHT))
        self.play(filter.animate.shift(cell_width * RIGHT))
        self.play(filter.animate.shift(cell_width * RIGHT))

        self.wait(2)


# Command to render the scene
if __name__ == "__main__":
    scene = KernelAnimation()
    scene.render()
