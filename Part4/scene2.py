from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene4_2(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Display Latent Space
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
            x_length=5,
            y_length=5,
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
        colorbar.arrange(DOWN, buff=0.2).to_edge(RIGHT, buff=2.5)

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

        self.play(FadeIn(axes), Write(x_label), Write(y_label))

        self.wait(0.6)

        self.wait(2)
        self.play(FadeIn(dots))
        self.play(FadeIn(colorbar), FadeIn(clabels))

        self.wait(0.7)

        graph = VGroup(axes, x_label, y_label, dots, colorbar, clabels)

        # Interpolate between two images
        self.next_section(skip_animations=False)

        self.wait(2)

        self.play(graph.animate.to_edge(RIGHT, buff=0.5))

        point0 = np.asarray([0.35, 0.13])
        point6 = np.asarray([0.75, 0.35])

        circle0 = Circle(radius=0.1, color=WHITE, stroke_width=2).move_to(
            axes.c2p(point0[0], point0[1])
        )

        circle6 = Circle(radius=0.1, color=WHITE, stroke_width=2).move_to(
            axes.c2p(point6[0], point6[1])
        )

        self.play(Create(circle0), run_time=1.5)
        self.play(Create(circle6), run_time=1.5)

        self.wait(0.7)

        original0 = (
            ImageMobject("images/interpolated/original0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(0.1)
        )
        original0.add(
            SurroundingRectangle(
                original0, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        ).move_to(circle0)
        original6 = (
            ImageMobject("images/interpolated/original6.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(0.1)
        )
        original6.add(
            SurroundingRectangle(
                original6, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        ).move_to(circle6)

        target_pos_0 = axes.get_left() + 2.5 * UP + 3 * LEFT
        target_pos_6 = axes.get_left() + 2.5 * DOWN + 3 * LEFT

        self.play(
            original0.animate.move_to(target_pos_0).scale(10 / 0.1),
            original6.animate.move_to(target_pos_6).scale(10 / 0.1),
        )

        img = (
            ImageMobject("images/interpolated/im0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        img.add(
            SurroundingRectangle(img, color=WHITE, stroke_width=2, buff=0.0, z_index=1)
        ).move_to(axes.get_left() + 3 * LEFT)

        self.play(FadeIn(img))

        trackerpoint = Dot(axes.c2p(point0[0], point0[1]), color=WHITE, radius=0.04)

        self.play(Create(trackerpoint))

        self.wait(0.4)

        coords = [
            ((1 - t) * point0 + t * point6).tolist() for t in np.linspace(0, 1, 60)
        ]

        for i in range(1, 60):
            new_img = (
                ImageMobject(f"images/interpolated/im{i}.png")
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .scale(10)
                .move_to(img)
            )
            new_img.add(
                SurroundingRectangle(
                    new_img, color=WHITE, stroke_width=2, buff=0.0, z_index=1
                )
            )

            self.remove(img)
            self.add(new_img)
            self.play(trackerpoint.animate.move_to(axes.c2p(*coords[i])), run_time=0.15)
            img = new_img

        self.wait(0.8)

        # Add noise and show reconstruction
        self.next_section(skip_animations=False)

        encoder = Polygon(
            [-1, 1, 0], [1, 0.4, 0], [1, -0.4, 0], [-1, -1, 0], color=PURPLE
        )
        bottleneck = Rectangle(width=1.5, height=0.8, color=BLUE)
        bottleneck.next_to(encoder, direction=RIGHT, buff=0.4)
        decoder = Polygon(
            [-1, 0.4, 0], [1, 1, 0], [1, -1, 0], [-1, -0.4, 0], color=PURPLE
        ).next_to(bottleneck, direction=RIGHT, buff=0.4)

        autoencoder = (
            VGroup(
                encoder,
                bottleneck,
                decoder,
            )
            .scale(0.9)
            .move_to(ORIGIN)
        )

        original1 = (
            ImageMobject("images/noise/original1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        original1.add(
            SurroundingRectangle(
                original1, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        )
        noisy1 = (
            ImageMobject("images/noise/noisy1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        noisy1.add(
            SurroundingRectangle(
                noisy1, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        ).move_to(axes.get_left() + 3 * LEFT)
        reconstructed1 = (
            ImageMobject("images/noise/reconstructed1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        reconstructed1.add(
            SurroundingRectangle(
                reconstructed1, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        )

        noisy1.next_to(autoencoder, LEFT, buff=1.0)
        reconstructed1.next_to(autoencoder, RIGHT, buff=1.0)
        original1.next_to(noisy1, DOWN, buff=0.5)

        txt_noisy = Tex("Noisy Image", color=WHITE).next_to(noisy1, UP, buff=0.5)
        txt_original = Tex("Original Image", color=WHITE).next_to(
            original1, RIGHT, buff=0.5
        )

        arrowin = Arrow(noisy1.get_right(), autoencoder.get_left(), color=WHITE)
        arrowout = Arrow(
            autoencoder.get_right(), reconstructed1.get_left(), color=WHITE
        )

        self.play(
            FadeOut(graph, circle0, circle6, original0, original6, img, trackerpoint)
        )

        self.play(FadeIn(autoencoder))

        self.wait(0.4)

        self.play(FadeIn(original1), Write(txt_original))
        self.play(FadeIn(noisy1), Write(txt_noisy))
        self.wait(0.5)

        self.play(GrowArrow(arrowin), run_time=1)
        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                FadeIn(reconstructed1),
                lag_ratio=0.5,
            )
        )

        self.wait(0.6)

        original4 = (
            ImageMobject("images/noise/original4.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        original4.add(
            SurroundingRectangle(
                original4, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        )
        noisy4 = (
            ImageMobject("images/noise/noisy4.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        noisy4.add(
            SurroundingRectangle(
                noisy4, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        ).move_to(axes.get_left() + 3 * LEFT)
        reconstructed4 = (
            ImageMobject("images/noise/reconstructed4.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        reconstructed4.add(
            SurroundingRectangle(
                reconstructed4, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        )

        noisy4.next_to(autoencoder, LEFT, buff=1.0)
        reconstructed4.next_to(autoencoder, RIGHT, buff=1.0)
        original4.next_to(noisy4, DOWN, buff=0.5)

        self.play(FadeOut(original1, noisy1, reconstructed1, arrowin, arrowout))

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeIn(noisy4),
                    FadeIn(original4),
                ),
                GrowArrow(arrowin),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                FadeIn(reconstructed4),
                lag_ratio=0.5,
            )
        )

        self.wait(0.5)

        original5 = (
            ImageMobject("images/noise/original5.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )

        original5.add(
            SurroundingRectangle(
                original5, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        )
        noisy5 = (
            ImageMobject("images/noise/noisy5.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        noisy5.add(
            SurroundingRectangle(
                noisy5, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        ).move_to(axes.get_left() + 3 * LEFT)
        reconstructed5 = (
            ImageMobject("images/noise/reconstructed5.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        reconstructed5.add(
            SurroundingRectangle(
                reconstructed5, color=WHITE, stroke_width=2, buff=0.0, z_index=1
            )
        )

        noisy5.next_to(autoencoder, LEFT, buff=1.0)
        reconstructed5.next_to(autoencoder, RIGHT, buff=1.0)
        original5.next_to(noisy5, DOWN, buff=0.5)

        self.play(FadeOut(original4, noisy4, reconstructed4, arrowin, arrowout))

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeIn(noisy5),
                    FadeIn(original5),
                ),
                GrowArrow(arrowin),
                lag_ratio=0.5,
            ),
            run_time=1,
        )

        self.play(
            LaggedStart(
                GrowArrow(arrowout),
                FadeIn(reconstructed5),
                lag_ratio=0.5,
            )
        )

        self.wait(0.7)

        # Conclusion
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(
                autoencoder,
                original5,
                noisy5,
                reconstructed5,
                arrowin,
                arrowout,
                txt_noisy,
                txt_original,
            )
        )

        self.wait(1)

        txt = Tex("Regularize the latent space", color=WHITE).scale(1.5)

        self.play(Write(txt))

        self.wait(0.6)

        self.play(FadeOut(txt))

        self.wait(1)

        txt = Tex("Variational Auto-Encoders", color=WHITE).scale(1.5)

        self.play(Write(txt))

        self.wait(3)

        self.play(FadeOut(txt))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene4_2()
    scene.render()
