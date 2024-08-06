from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene2_2(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Draw basic autoencoder
        self.next_section(skip_animations=False)

        encoder = Polygon(
            [-1, 1, 0], [1, 0.4, 0], [1, -0.4, 0], [-1, -1, 0], color=PURPLE
        )
        encoder_txt = (
            Tex("Encoder", color=WHITE).scale(0.6).move_to(encoder.get_center())
        )

        bottleneck_txt = Tex("Bottleneck", color=WHITE).scale(0.6)
        bottleneck = SurroundingRectangle(bottleneck_txt, buff=0.3, color=BLUE)
        bottleneck_g = VGroup(bottleneck, bottleneck_txt)
        bottleneck_g.next_to(encoder, direction=RIGHT, buff=0.4)

        decoder = Polygon(
            [-1, 0.4, 0], [1, 1, 0], [1, -1, 0], [-1, -0.4, 0], color=PURPLE
        ).next_to(bottleneck, direction=RIGHT, buff=0.4)
        decoder_txt = (
            Tex("Decoder", color=WHITE).scale(0.6).move_to(decoder.get_center())
        )

        autoencoder = VGroup(
            encoder, bottleneck, decoder, encoder_txt, bottleneck_txt, decoder_txt
        )
        autoencoder.move_to(ORIGIN).scale(0.9)

        input_img1 = (
            ImageMobject("images/mnist/original_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        input_img1.next_to(autoencoder, direction=LEFT, buff=1.0)
        input_img1_rect = SurroundingRectangle(
            input_img1, buff=0.0, color=WHITE, stroke_width=1
        )
        input_img1_title = (
            Tex("Input", color=WHITE)
            .scale(0.6)
            .next_to(input_img1, direction=UP, buff=0.3)
        )

        output_img1 = (
            ImageMobject("images/mnist/reconstructed_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        output_img1.next_to(autoencoder, direction=RIGHT, buff=1.0)
        output_img1_rect = SurroundingRectangle(
            output_img1, buff=0.0, color=WHITE, stroke_width=1
        )
        output_img1_title = (
            Tex("Output", color=WHITE)
            .scale(0.6)
            .next_to(output_img1, direction=UP, buff=0.3)
        )

        input_img1_g = Group(input_img1, input_img1_rect, input_img1_title)
        output_img1_g = Group(output_img1, output_img1_rect, output_img1_title)

        arrowin = Arrow(input_img1.get_right(), autoencoder.get_left(), color=WHITE)
        arrowout = Arrow(autoencoder.get_right(), output_img1.get_left(), color=WHITE)

        self.play(FadeIn(input_img1_g), run_time=1)
        self.play(GrowArrow(arrowin), run_time=0.5)
        self.play(FadeIn(autoencoder), run_time=2)
        self.play(GrowArrow(arrowout), run_time=0.5)
        self.play(FadeIn(output_img1_g), run_time=1)

        self.wait(0.5)

        # Focus on decoder

        ax = (
            Axes(
                x_range=[0, 1],
                y_range=[0, 1],
                x_length=3,
                y_length=3,
                axis_config={
                    "color": WHITE,
                    "include_tip": True,
                    "numbers_to_include": np.arange(0, 1, 1),
                    "tip_height": 0.2,
                    "tip_width": 0.2,
                },
            )
            .scale(0.9)
            .next_to(decoder, direction=LEFT, buff=1.0)
        )

        labels = ax.get_axis_labels(x_label="Feature 1", y_label="Feature 2")
        labels.scale(0.6)
        labels[0].shift(0.4 * DOWN + 0.2 * LEFT)
        labels[1].shift(0.8 * LEFT)
        dot_human = Dot(ax.c2p(0.25, 0.4), color=WHITE)
        arrow = Arrow(ax.get_right(), decoder.get_left(), color=WHITE)

        self.play(
            LaggedStart(
                FadeOut(
                    arrowin,
                    encoder,
                    encoder_txt,
                    bottleneck,
                    bottleneck_txt,
                    input_img1_g,
                ),
                FadeIn(ax, labels, dot_human),
                GrowArrow(arrow),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.wait(0.4)

        # Focus on encoder
        self.play(
            LaggedStart(
                FadeOut(arrow, decoder, decoder_txt, output_img1_g, arrowout),
                VGroup(ax, labels, dot_human).animate.shift(3 * RIGHT),
                lag_ratio=0.5,
            ),
            run_time=2,
        )
        arrow2 = Arrow(encoder.get_right(), ax.get_left(), color=WHITE)

        self.play(
            LaggedStart(
                FadeIn(encoder, encoder_txt, input_img1_g, arrowin),
                GrowArrow(arrow2),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        self.play(
            FadeOut(ax, labels, dot_human, arrow2),
            FadeIn(
                output_img1_g,
                bottleneck,
                bottleneck_txt,
                decoder,
                decoder_txt,
                arrowout,
            ),
            run_time=2,
        )

        # How to compare images
        txt = Tex("How to compare images ?", color=WHITE)
        txt.to_edge(DOWN, buff=1.5)
        self.play(Write(txt), run_time=1)

        self.wait(0.3)

        # Zoom in on the images and compare 2 pixels
        self.next_section(skip_animations=False)

        input_img1_g = Group(input_img1, input_img1_rect)
        output_img1_g = Group(output_img1, output_img1_rect)

        input_img1_values = plt.imread("images/mnist/original_2.png")
        output_img1_values = plt.imread("images/mnist/reconstructed_2.png")

        pixel1_value = input_img1_values[7, 14] * 255
        pixel2_value = output_img1_values[7, 14] * 255

        self.play(FadeOut(txt), run_time=1)

        self.play(
            LaggedStart(
                FadeOut(arrowin, arrowout, autoencoder),
                AnimationGroup(
                    *[
                        FadeOut(input_img1_title, output_img1_title),
                        input_img1_g.animate.scale(1.5).shift(3 * RIGHT + 2 * UP),
                        output_img1_g.animate.scale(1.5).shift(3 * LEFT + 2 * UP),
                    ]
                ),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        cell_width = input_img1.width / 28
        pixel1 = Rectangle(
            width=cell_width,
            height=cell_width,
            color=BLUE,
            fill_color=ManimColor(pixel1_value),
            fill_opacity=1,
            stroke_width=2,
        ).move_to(
            input_img1_g.get_corner(UL)
            + 0.5 * cell_width * (RIGHT + DOWN)
            + 14 * cell_width * RIGHT
            + 7 * cell_width * DOWN
        )
        pixel2 = Rectangle(
            width=cell_width,
            height=cell_width,
            color=RED,
            fill_color=ManimColor(pixel2_value),
            fill_opacity=1,
            stroke_width=2,
        ).move_to(
            output_img1_g.get_corner(UL)
            + 0.5 * cell_width * (RIGHT + DOWN)
            + 14 * cell_width * RIGHT
            + 7 * cell_width * DOWN
        )

        self.wait(0.5)

        self.play(Create(pixel1), Create(pixel2), run_time=2)

        self.play(
            pixel1.animate.shift(4 * DOWN + RIGHT).scale(5),
            pixel2.animate.shift(4 * DOWN + LEFT).scale(5),
            run_time=1,
        )

        self.wait(0.5)

        minus = Tex("-", color=WHITE).move_to(
            0.5 * (pixel1.get_center() + pixel2.get_center())
        )
        self.play(Write(minus), run_time=1)

        p1_txt = Tex(f"{pixel1_value[0]:.0f}", color=WHITE).move_to(pixel1.get_center())
        p2_txt = Tex(f"{pixel2_value[0]:.0f}", color=WHITE).move_to(pixel2.get_center())
        res = pixel1_value - pixel2_value
        self.play(Transform(pixel1, p1_txt), Transform(pixel2, p2_txt), run_time=2)
        self.wait(0.5)
        result_txt = Tex(f"{res[0]}", color=WHITE).move_to(minus.get_center())
        exp = VGroup(pixel1, pixel2, minus)
        self.play(Transform(exp, result_txt), run_time=1)

        self.wait(0.6)

        # Display the loss
        self.next_section(skip_animations=False)

        loss_function_title = (
            Tex("Loss function", color=WHITE).shift(0.5 * DOWN).scale(0.8)
        )
        loss_function_underline = Underline(loss_function_title, buff=0.1)
        loss_function = MathTex(
            r"\mathcal{L}(x, y) = \frac{1}{n} \sum_{i=1}^{n}",
            r"(",
            r"x_i - y_i",
            r")",
            r"^2",
            color=WHITE,
        ).next_to(loss_function_title, direction=DOWN, buff=0.5)

        self.play(FadeOut(exp), run_time=1)

        self.play(
            Write(loss_function_title),
            GrowFromEdge(loss_function_underline, LEFT),
            run_time=1,
        )
        self.play(FadeIn(loss_function), run_time=2)

        self.wait(0.3)

        o = VGroup(loss_function[1], loss_function[3], loss_function[4])
        self.play(Indicate(o, color=RED), run_time=1)
        self.play(Indicate(o, color=RED), run_time=1)

        self.wait(0.5)

        self.play(
            FadeOut(loss_function_title, loss_function, loss_function_underline),
            run_time=1,
        )
        txt = Tex("Loss: 0.0456", color=WHITE).scale(0.8).shift(2 * DOWN)
        self.play(Write(txt), run_time=2)
        self.play(FadeOut(txt), run_time=1)

        self.wait(0.7)

        # Display the autoencoder and the initial images
        self.next_section(skip_animations=False)

        self.play(
            input_img1_g.animate.shift(3 * LEFT).scale(1 / 1.5),
            output_img1_g.animate.shift(3 * RIGHT).scale(1 / 1.5),
            run_time=1,
        )
        autoencoder.shift(2 * UP)
        arrowin.shift(2 * UP)
        arrowout.shift(2 * UP)

        self.play(FadeIn(autoencoder, arrowin, arrowout), run_time=1)

        self.wait(0.6)

        input_img2 = (
            ImageMobject("images/mnist/progress/ref_image.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        input_img2_rect = SurroundingRectangle(
            input_img2, buff=0.0, color=WHITE, stroke_width=1
        )

        output_img_progress = (
            ImageMobject("images/mnist/progress/epoch_1_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        output_img_progress_rect = SurroundingRectangle(
            output_img_progress, buff=0.0, color=WHITE, stroke_width=1
        )
        output_img_progress_rect.z_index = 1

        input_img2_g = Group(input_img2, input_img2_rect).move_to(input_img1_g)
        output_img_progress_g = Group(
            output_img_progress, output_img_progress_rect
        ).move_to(output_img1_g)

        # Plot the loss curve and display the images
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(input_img1_g, output_img1_g),
            run_time=1,
        )

        self.play(FadeIn(input_img2_g, output_img_progress_g), run_time=1)

        values = np.load("values/losses.npy")

        # Convert values to a continuous function (interpolated function)
        x_values = np.arange(0, 20, 1 / 5)

        y_values = values

        # Create the axes
        axes = Axes(
            x_range=[0, max(x_values), 1],
            y_range=[0, max(values) + 0.01, 0.01],
            x_length=10,
            y_length=5,
            x_axis_config={
                "color": WHITE,
                "tip_height": 0.25,
                "tip_width": 0.25,
                "include_numbers": True,
                "numbers_to_include": np.arange(0, max(x_values) + 1, 5),
            },
            y_axis_config={
                "color": WHITE,
                "tip_height": 0.25,
                "tip_width": 0.25,
                "include_numbers": True,
                "numbers_to_include": np.arange(0, max(y_values), 0.01),
            },
        )

        axes.shift(2 * DOWN).scale(0.7)
        axes_labels = axes.get_axis_labels(x_label="Epochs", y_label="Loss")

        # Add axes and labels to the scene
        self.play(Create(axes), FadeIn(axes_labels))

        self.wait(0.3)

        # Initialize the line graph
        graph = VMobject()

        # Create the line graph

        for i in range(len(x_values) - 1):
            line = Line(
                start=axes.c2p(x_values[i], y_values[i]),
                end=axes.c2p(x_values[i + 1], y_values[i + 1]),
                color=BLUE,
                stroke_width=2,
            )
            graph.add(line)

        # Add the line graph to the scene

        img_target_prev = output_img_progress

        for i in range(len(graph)):
            epoch = i // 5 + 1
            idx = i % 5 + 1
            path = f"images/mnist/progress/epoch_{epoch}_{idx}.png"
            img_target = (
                ImageMobject(path)
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .move_to(output_img_progress)
                .scale(10)
            )

            self.play(Create(graph[i]), run_time=0.1)
            self.remove(img_target_prev)
            self.add(img_target)
            img_target_prev = img_target

        self.wait(0.5)

        self.next_section(skip_animations=False)

        self.play(
            FadeOut(
                input_img2_g,
                img_target,
                output_img_progress_rect,
                graph,
                axes,
                axes_labels,
                autoencoder,
                arrowin,
                arrowout,
            ),
            run_time=2,
        )

        txt = Tex("Why use autoencoders ?", color=WHITE).scale(1.5)
        txt_underline = Underline(txt, buff=0.1)

        self.play(Write(txt), GrowFromEdge(txt_underline, LEFT), run_time=1)

        self.wait(1)

        self.play(FadeOut(txt, txt_underline), run_time=1)

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_2()
    scene.render()
