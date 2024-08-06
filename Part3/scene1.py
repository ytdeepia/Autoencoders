from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene3_1(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Display the autoencoder and the number of output neurons
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

        self.play(FadeIn(autoencoder))

        title = (
            Tex("Dimensionality Reduction", color=WHITE).scale(0.8).to_edge(UP, buff=1)
        )
        self.wait()
        self.play(Write(title))

        self.wait(0.5)

        neurons = VGroup(
            Circle(radius=0.45, color=WHITE),
            Circle(radius=0.45, color=WHITE),
        )

        neurons.arrange(DOWN, buff=0.4).next_to(encoder, RIGHT, buff=1.5)
        line1 = Line(
            encoder.get_corner(UR) + 0.4 * RIGHT + 0.5 * DOWN,
            neurons[0].get_left(),
            color=WHITE,
        )
        line2 = Line(
            encoder.get_corner(DR) + 0.4 * RIGHT + 0.5 * UP,
            neurons[0].get_left(),
            color=WHITE,
        )
        line3 = Line(
            encoder.get_corner(UR) + 0.4 * RIGHT + 0.5 * DOWN,
            neurons[1].get_left(),
            color=WHITE,
        )
        line4 = Line(
            encoder.get_corner(DR) + 0.4 * RIGHT + 0.5 * UP,
            neurons[1].get_left(),
            color=WHITE,
        )

        neuron1_txt = (
            Tex("Neuron 1", color=WHITE)
            .scale(0.5)
            .next_to(neurons[0], direction=UP, buff=0.2)
        )
        neuron2_txt = (
            Tex("Neuron 2", color=WHITE)
            .scale(0.5)
            .next_to(neurons[1], direction=DOWN, buff=0.2)
        )

        self.play(FadeOut(title))
        self.wait()
        self.play(FadeOut(decoder, decoder_txt), run_time=2)
        self.play(
            Indicate(
                bottleneck,
                scale_factor=1.2,
                color=bottleneck.get_color(),
            ),
            Indicate(
                bottleneck_txt,
                scale_factor=1.2,
                color=bottleneck_txt.get_color(),
            ),
        )

        self.wait(0.3)

        fc_layer = VGroup(line1, line2, line3, line4, neurons)

        self.wait()
        self.play(
            LaggedStart(FadeOut(bottleneck_g), FadeIn(fc_layer), lag_ratio=0.5),
            run_time=2,
        )
        self.play(Write(neuron1_txt), Write(neuron2_txt))

        fc_layer.add(neuron1_txt, neuron2_txt)

        self.wait(0.6)

        # Demonstrate how to visualize the latent space
        self.next_section(skip_animations=False)
        axes = (
            Axes(
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
            .to_edge(RIGHT, buff=1.5)
            .shift(0.5 * DOWN)
        )

        x_label, y_label = axes.get_axis_labels(
            x_label="Neuron 1", y_label="Neuron 2"
        ).scale(0.4)
        x_label.move_to(axes.c2p(1, 0) + (0.2 + x_label.height) * UP)
        y_label.move_to(axes.c2p(0, 1) + (0.2 + y_label.width / 2) * RIGHT)
        labels = VGroup(x_label, y_label)

        title = (
            Tex("Latent Space", color=WHITE)
            .scale(0.8)
            .next_to(axes, direction=UP, buff=0.8)
        )
        title_underline = Underline(title, buff=0.1, color=WHITE)

        autoencoder2 = VGroup(encoder, encoder_txt, fc_layer)
        axes_g = VGroup(axes, x_label, y_label)
        axes_g.to_edge(RIGHT, buff=1)

        self.play(autoencoder2.animate.shift(3 * LEFT))
        self.play(Create(axes), run_time=2)
        self.wait(1)
        self.play(Write(x_label))
        self.play(Write(y_label))
        self.play(FadeIn(title, title_underline))

        self.wait(0.4)

        dot = Dot(axes.c2p(0.8, 0.2), color=RED, radius=0.05, fill_opacity=0.8)
        self.play(Indicate(neurons[0], scale_factor=1.2, color=WHITE))
        self.wait(0.5)
        self.play(ApplyWave(x_label, amplitude=0.2), run_time=2)
        self.wait(0.3)
        self.play(Create(dot))

        self.wait(0.3)

        self.play(Indicate(neurons[1], scale_factor=1.2, color=WHITE))
        self.wait(0.5)
        self.play(ApplyWave(y_label, amplitude=0.2), run_time=2)
        self.wait(0.3)
        self.play(dot.animate.move_to(axes.c2p(0.8, 0.8)))

        self.wait(0.4)

        # Number being encoded
        self.next_section(skip_animations=False)

        bottleneck_g.next_to(encoder, direction=RIGHT, buff=0.4)

        self.play(
            LaggedStart(
                FadeOut(fc_layer),
                FadeIn(bottleneck_g),
                lag_ratio=0.5,
            ),
            run_time=2,
        )

        autoencoder = VGroup(encoder, bottleneck, encoder_txt, bottleneck_txt)
        self.play(autoencoder.animate.to_edge(LEFT, buff=1.5).scale(0.5))

        input_img1 = (
            ImageMobject("images/mnist/latent_instances/7/image_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(6)
        )
        input_img1_rect = SurroundingRectangle(
            input_img1, buff=0.0, color=WHITE, stroke_width=1
        )
        input_img1_rect.z_index = 1
        input_img1_g = Group(input_img1, input_img1_rect)
        input_img1_g.next_to(autoencoder, direction=LEFT, buff=0.8)
        arrowin = Arrow(
            input_img1_g.get_right(),
            autoencoder.get_left(),
            buff=0.1,
            color=WHITE,
            max_tip_length_to_length_ratio=0.15,
        )

        self.play(
            LaggedStart(FadeIn(input_img1_g), GrowArrow(arrowin), lag_ratio=0.4),
            run_time=2,
        )

        dots_7 = VGroup()

        self.wait(0.2)

        # Display several instances of the same number being encoded
        self.next_section(skip_animations=False)

        dot1 = Dot(axes.c2p(0.8, 0.8), color=RED, radius=0.04, fill_opacity=0.8)
        dots_7.add(dot1)
        self.play(Create(dot1))

        input_img_prev = input_img1

        coord_x = [0.8, 0.85, 0.78, 0.82, 0.87]
        coord_y = [0.8, 0.82, 0.85, 0.79, 0.84]

        self.play(FadeOut(dot))

        for i in range(1, 5):
            input_img_next = (
                ImageMobject(f"images/mnist/latent_instances/7/image_{i}.png")
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .scale(6)
            ).move_to(input_img1)

            dot = Dot(
                axes.c2p(
                    coord_x[i],
                    coord_y[i],
                ),
                color=RED,
                radius=0.04,
                fill_opacity=0.8,
            )
            dots_7.add(dot)
            self.play(
                LaggedStart(
                    FadeOut(input_img_prev),
                    FadeIn(input_img_next),
                    lag_ratio=0.4,
                ),
                run_time=0.8,
            )

            self.play(Create(dot))

            self.wait(0.2)

            input_img_prev = input_img_next

        # Display examples from other classes
        self.next_section(skip_animations=False)

        coord_x = [0.1, 0.12, 0.08, 0.15, 0.11]
        coord_y = [0.1, 0.12, 0.15, 0.09, 0.13]
        dots_4 = VGroup()

        for i in range(0, 5):
            input_img_next = (
                ImageMobject(f"images/mnist/latent_instances/4/image_{i}.png")
                .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
                .scale(6)
            ).move_to(input_img1)

            dot = Dot(
                axes.c2p(
                    coord_x[i],
                    coord_y[i],
                ),
                color=GREEN,
                radius=0.04,
                fill_opacity=0.8,
            )

            self.play(
                LaggedStart(
                    FadeOut(input_img_prev),
                    FadeIn(input_img_next),
                    lag_ratio=0.4,
                ),
                run_time=0.8,
            )
            dots_4.add(dot)

            self.play(Create(dot))

            self.wait(0.2)

            input_img_prev = input_img_next

        circle_7 = Circle(color=RED).surround(dots_7)
        circle_4 = Circle(color=GREEN).surround(dots_4)

        txt7 = Tex("7", color=RED).scale(0.5).next_to(circle_7, direction=UP, buff=0.2)
        txt4 = (
            Tex("4", color=GREEN).scale(0.5).next_to(circle_4, direction=UP, buff=0.2)
        )

        self.play(Create(circle_7), FadeIn(txt7))
        self.wait(0.3)
        self.play(Create(circle_4), FadeIn(txt4))

        # Show how this representation evolves during training
        self.next_section(skip_animations=False)

        self.play(
            FadeOut(
                dots_4,
                circle_4,
                txt4,
                dots_7,
                circle_7,
                txt7,
                input_img1_rect,
                input_img_next,
                arrowin,
                encoder,
                bottleneck,
                encoder_txt,
                bottleneck_txt,
                title,
                title_underline,
            ),
            run_time=2,
        )
        axes_g = VGroup(axes, labels)
        self.play(axes_g.animate.move_to(ORIGIN))

        # Colorbar on the right
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

        colorbar.arrange(DOWN, buff=0.2).to_edge(RIGHT, buff=1.5)
        for idx, clabel in enumerate(clabels):
            clabel.next_to(colorbar[idx], direction=RIGHT, buff=0.3)

        self.play(FadeIn(colorbar), FadeIn(clabels))

        # Epoch counter
        epoch_counter = Tex("Epoch 1", color=WHITE).scale(0.9).to_edge(UP, buff=0.5)
        epoch_counter.to_corner(UL, buff=1)

        self.play(FadeIn(epoch_counter))

        # Training animation
        min_latent = [-56.432663, -51.0675]
        max_latent = [8.780535, 7.984861]

        latent_space = np.load(f"images/latent_spaces_training/0/latent_space.npy")
        labels = np.load(f"images/latent_spaces_training/0/labels.npy")

        dots = VGroup()

        for p in range(len(labels)):
            dot = Dot(
                axes.c2p(
                    (latent_space[p][0] - min_latent[0])
                    / (max_latent[0] - min_latent[0]),
                    (latent_space[p][1] - min_latent[1])
                    / (max_latent[1] - min_latent[1]),
                ),
                color=colors[labels[p]],
                fill_opacity=0.8,
                radius=0.03,
            )
            dots.add(dot)

        self.play(FadeIn(dots), run_time=1)

        for epoch in range(1, 10):
            latent_space = np.load(
                f"images/latent_spaces_training/{epoch}/latent_space.npy"
            )
            labels = np.load(f"images/latent_spaces_training/{epoch}/labels.npy")

            epoch_counter_target = (
                Tex(f"Epoch {epoch+1}", color=WHITE).scale(0.9)
            ).move_to(epoch_counter)

            new_positions = [
                axes.c2p(
                    (latent_space[p][0] - min_latent[0])
                    / (max_latent[0] - min_latent[0]),
                    (latent_space[p][1] - min_latent[1])
                    / (max_latent[1] - min_latent[1]),
                )
                for p in range(len(labels))
            ]

            animations = [
                dot.animate.move_to(new_pos)
                for dot, new_pos in zip(dots, new_positions)
            ]

            self.play(
                Transform(epoch_counter, epoch_counter_target),
                *animations,
                run_time=0.7,
            )
            self.wait(0.1)

        for epoch in range(10, 20):
            latent_space = np.load(
                f"images/latent_spaces_training/{epoch}/latent_space.npy"
            )
            labels = np.load(f"images/latent_spaces_training/{epoch}/labels.npy")

            epoch_counter_target = (
                Tex(f"Epoch {epoch+1}", color=WHITE).scale(0.9)
            ).move_to(epoch_counter)

            new_positions = [
                axes.c2p(
                    (latent_space[p][0] - min_latent[0])
                    / (max_latent[0] - min_latent[0]),
                    (latent_space[p][1] - min_latent[1])
                    / (max_latent[1] - min_latent[1]),
                )
                for p in range(len(labels))
            ]

            animations = [
                dot.animate.move_to(new_pos)
                for dot, new_pos in zip(dots, new_positions)
            ]

            self.play(
                Transform(epoch_counter, epoch_counter_target),
                *animations,
                run_time=0.7,
            )
            self.wait(0.1)

        # Fade Out
        self.next_section(skip_animations=False)

        self.play(FadeOut(dots, epoch_counter, axes_g, colorbar, clabels))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_1()
    scene.render()
