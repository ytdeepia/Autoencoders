from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene3_4(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Quality of latent space comparison
        self.next_section(skip_animations=False)

        colors = [
            RED,
            GREEN,
            BLUE,
            YELLOW,
            PURPLE,
            ORANGE,
            PINK,
            TEAL,
            DARK_BROWN,
            GREY,
        ]

        axes = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 1, 0.5],
            x_length=6,
            y_length=6,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 18,
            },
        )

        x_label = (
            Text("Dimension 1", font_size=16)
            .next_to(axes.x_axis, DOWN)
            .shift((axes.width / 2) * RIGHT)
        )
        y_label = (
            Text("Dimension 2", font_size=16)
            .next_to(axes.y_axis, LEFT)
            .shift((axes.height / 2) * UP)
        )

        latent_space = np.load(f"images/latent_space2D/latent_space.npy")
        labels = np.load(f"images/latent_space2D/labels.npy")

        colorbar = VGroup()
        clabels = VGroup()
        for i, color in enumerate(colors):
            rect = Rectangle(
                width=0.6,
                height=0.3,
                color=color,
                fill_opacity=1,
                stroke_width=0,
            )
            label = Tex(f"{i}", color=WHITE).scale(0.4)
            colorbar.add(rect)
            clabels.add(label)
        colorbar.next_to(axes, direction=RIGHT, buff=0.5)
        colorbar.arrange(DOWN, buff=0.2).to_edge(RIGHT, buff=1.5)
        for idx, clabel in enumerate(clabels):
            clabel.next_to(colorbar[idx], direction=RIGHT, buff=0.3)

        dots = VGroup()

        for p in range(len(labels)):
            dot = Dot(
                axes.c2p(
                    latent_space[p][0],
                    latent_space[p][1],
                ),
                color=colors[labels[p]],
                fill_opacity=0.8,
                radius=0.03,
            )
            dots.add(dot)

        title = Tex("Latent dimension = ", "2").to_edge(UP, buff=0.25)
        title_underline = Underline(title, buff=0.1)

        self.play(Create(axes), run_time=2)
        self.play(FadeIn(x_label, y_label, colorbar, clabels), run_time=2)
        self.play(FadeIn(title), GrowFromEdge(title_underline, LEFT), run_time=1)
        self.play(FadeIn(dots), run_time=2)

        self.wait(0.5)

        self.wait(0.7)

        latent_space = np.load(f"images/latent_space5D/latent_space.npy")
        labels = np.load(f"images/latent_space5D/labels.npy")

        new_positions = [
            axes.c2p(
                latent_space[p][0],
                latent_space[p][1],
            )
            for p in range(len(labels))
        ]

        animations = [
            dot.animate.move_to(new_pos) for dot, new_pos in zip(dots, new_positions)
        ]

        title2 = Tex("Latent dimension = ", "5").to_edge(UP, buff=0.25)

        self.play(
            *animations,
            LaggedStart(
                FadeOut(title),
                FadeIn(title2),
                lag_ratio=0.2,
                run_time=1,
            ),
            run_time=5,
        )

        self.remove(title)

        self.wait(0.7)

        # Indicate cluster of classes
        self.next_section(skip_animations=False)

        ellipse_1 = Ellipse(width=2.8, height=1.2, color=colors[1], stroke_width=2)
        ellipse_1.move_to(axes.c2p(0.7, 0.9)).rotate(-20 * DEGREES)
        label_1 = Tex("1", color=colors[1]).next_to(ellipse_1, LEFT, buff=0.2)
        ellipse_2 = Ellipse(width=2, height=1, color=colors[2], stroke_width=2)
        ellipse_2.move_to(axes.c2p(0.38, 0.56)).rotate(40 * DEGREES)
        label_2 = Tex("2", color=colors[2]).next_to(ellipse_2, UP, buff=0.2)
        ellispe_7 = Ellipse(width=2.6, height=1.3, color=colors[7], stroke_width=2)
        ellispe_7.move_to(axes.c2p(0.8, 0.73)).rotate(-10 * DEGREES)
        label_7 = Tex("7", color=colors[7]).next_to(ellispe_7, DOWN, buff=0.2)

        self.play(Create(ellipse_1), run_time=1)
        self.play(Write(label_1), run_time=1)
        self.wait(1.5)
        self.play(Create(ellipse_2), run_time=1)
        self.play(Write(label_2), run_time=1)
        self.wait(1.5)
        self.play(Create(ellispe_7), run_time=1)
        self.play(Write(label_7), run_time=1)

        self.play(
            FadeOut(ellipse_1, ellipse_2, ellispe_7, label_1, label_2, label_7),
            run_time=1,
        )

        # How do we visualize 3D and more into 2D ?
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(
                axes,
                dots,
                x_label,
                y_label,
                colorbar,
                clabels,
                title2,
                title_underline,
            ),
            run_time=1,
        )

        txt = Tex("Stay tuned for the next video !").scale(1.5)
        self.play(Write(txt), run_time=1)

        self.play(FadeOut(txt), run_time=1)

        # Instances of classes being missreconstructed
        self.next_section(skip_animations=False)

        point_src = axes.get_left() + 2 * LEFT + 1.5 * UP
        point_target = axes.get_left() + 2 * LEFT + 2 * DOWN

        start_point = axes.c2p(0.16, 0.64)
        circle = Circle(radius=0.1, color=WHITE, stroke_width=2).move_to(start_point)

        image_2 = (
            ImageMobject("images/missreconstructions/image_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )

        recon_2 = (
            ImageMobject("images/missreconstructions/recon_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(0.1)
        )

        image_2.move_to(point_src)
        recon_2.move_to(start_point)

        final_scale = 10.0

        self.play(
            FadeIn(
                axes,
                dots,
                x_label,
                y_label,
                colorbar,
                clabels,
                title2,
                title_underline,
            )
        )

        self.wait(0.4)

        self.play(FadeIn(image_2), run_time=2)
        self.play(Create(circle), run_time=1)
        self.play(Indicate(circle, color=WHITE), run_time=1)

        self.add(image_2)

        # Animate the image to grow and move towards the target point
        self.play(
            recon_2.animate.move_to(point_target).scale(final_scale / 0.1),
            run_time=3,  # Duration of the animation
        )

        self.play(FadeOut(circle, image_2, recon_2), run_time=0.7)

        start_point = axes.c2p(0.8, 0.2)
        circle = Circle(radius=0.1, color=WHITE, stroke_width=2).move_to(start_point)

        image_3 = (
            ImageMobject("images/missreconstructions/image_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )

        recon_3 = (
            ImageMobject("images/missreconstructions/recon_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(0.1)
        )

        image_3.move_to(point_src)
        recon_3.move_to(start_point)

        self.play(FadeIn(image_3), run_time=2)
        self.play(Create(circle), run_time=1)
        self.play(Indicate(circle, color=WHITE), run_time=1)

        self.add(image_3)

        self.play(
            recon_3.animate.move_to(point_target).scale(final_scale / 0.1),
            run_time=3,
        )

        self.play(
            FadeOut(
                circle,
                image_3,
                recon_3,
                axes,
                colorbar,
                clabels,
                title2,
                title_underline,
                dots,
                x_label,
                y_label,
            ),
            run_time=2,
        )

        txt = Tex("Let's see some applications !").scale(1.5)

        self.play(Write(txt), run_time=1)

        self.play(FadeOut(txt), run_time=1)

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_4()
    scene.render()
