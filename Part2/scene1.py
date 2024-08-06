from manim import *
import numpy as np
from manim_voiceover import VoiceoverScene


class Scene2_1(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Title
        self.next_section(skip_animations=False)

        title = Text("Autoencoders", color=WHITE).scale(1)
        self.play(AddTextLetterByLetter(title, time_per_char=0.1))
        underline = Underline(title, buff=0.1)
        self.play(GrowFromEdge(underline, LEFT))
        title = VGroup(title, underline)

        self.wait(0.5)

        num_rows = 5
        num_cols = 5
        random_numbers = np.random.randint(1, 100, size=(num_rows, num_cols))
        random_matrix = [[str(num) for num in row] for row in random_numbers]
        random_matrix.append([r"\dots" for _ in range(num_cols)])
        large_matrix = Matrix(random_matrix)
        large_matrix.scale(0.7)
        large_matrix_title = Tex("Data").scale(0.6).next_to(large_matrix, UP)

        random_numbers = np.random.randint(1, 100, size=(3, 1))
        features = [[str(num) for num in row] for row in random_numbers]
        features = Matrix(features)
        features.scale(0.7).shift(3 * RIGHT)
        features_title = Tex("Features").scale(0.6).next_to(features, UP)

        self.play(FadeOut(title), run_time=1)
        self.play(Create(large_matrix), FadeIn(large_matrix_title), run_time=2)
        self.play(VGroup(large_matrix, large_matrix_title).animate.shift(3 * LEFT))
        arrow = Arrow(large_matrix.get_right(), features.get_left(), buff=0.4)

        self.play(
            LaggedStart(
                GrowArrow(arrow),
                AnimationGroup(*[Create(features), FadeIn(features_title)]),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        self.wait(0.7)

        # Human comparison
        self.next_section(skip_animations=False)

        human = SVGMobject(
            "images/human.svg",
            opacity=1,
            fill_opacity=1,
            stroke_color=WHITE,
            fill_color=WHITE,
        ).scale(1.5)

        brace_human = Brace(human, direction=RIGHT, buff=0.1)
        brace_human_txt = brace_human.get_text("1m80 / 6 ft.").scale(0.6)

        age_human = (
            Tex("Age: 25", color=WHITE)
            .scale(0.6)
            .next_to(human, direction=LEFT, buff=0.5)
        )

        self.play(
            FadeOut(large_matrix, large_matrix_title, features, features_title, arrow)
        )
        self.play(FadeIn(human), run_time=2)

        self.wait(0.3)

        self.play(FadeIn(brace_human))
        self.play(Write(brace_human_txt))
        self.play(Write(age_human))

        self.wait(0.4)

        ax = Axes(
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
        ).shift(2 * RIGHT)

        labels = ax.get_axis_labels(x_label="Age", y_label="Height")
        dot_human = Dot(ax.c2p(0.25, 0.8), color=WHITE)

        self.play(
            LaggedStart(
                FadeOut(brace_human, brace_human_txt, age_human),
                human.animate.shift(2 * LEFT),
                lag_ratio=0.4,
            ),
            run_time=2,
        )

        self.play(Create(ax), Write(labels), run_time=2)
        self.play(Create(dot_human))

        self.wait(0.7)

        self.play(FadeOut(human, ax, labels, dot_human))

        self.wait(0.7)

        # Draw basic autoencoder architecture
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

        self.play(Create(encoder))
        self.play(Write(encoder_txt))
        self.play(Create(bottleneck))
        self.play(Write(bottleneck_txt))
        self.play(Create(decoder))
        self.play(Write(decoder_txt))

        self.wait(0.7)

        # Draw the input and output data
        self.next_section(skip_animations=False)

        input_data = Matrix(
            [[0.11, 0.78], [0.45, 0.52]], left_bracket="[", right_bracket="]"
        ).scale(0.8)

        output_data = input_data.copy()

        input_data.next_to(autoencoder, direction=LEFT, buff=1.5)
        output_data.next_to(autoencoder, direction=RIGHT, buff=1.5)
        arrow_in = Arrow(input_data.get_right(), autoencoder.get_left(), buff=0.2)
        arrow_out = Arrow(autoencoder.get_right(), output_data.get_left(), buff=0.2)

        self.play(FadeIn(input_data))
        self.play(GrowArrow(arrow_in))
        self.play(Indicate(encoder, color=encoder.color))

        self.wait(0.7)

        self.play(
            autoencoder.animate.shift(2 * UP),
            input_data.animate.shift(2 * UP),
            arrow_in.animate.shift(2 * UP),
        )

        arrow_out.shift(2 * UP)
        output_data.shift(2 * UP)

        ax = Axes(
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
        labels = ax.get_axis_labels(x_label="z1", y_label="z2")
        dot1 = Dot(ax.c2p(0.3, 0.8), color=RED)

        latent_space = VGroup(ax, labels, dot1)
        latent_space.to_edge(DOWN, buff=0.8)

        dot1_legend = (
            Tex("Encoded data", color=RED)
            .scale(0.6)
            .next_to(dot1, direction=DOWN, buff=0.2)
        )

        ax_title = Tex("Latent space", color=WHITE).scale(0.6)
        ax_title.next_to(ax, direction=DOWN, buff=0.2)
        ax_title_underline = Underline(ax_title, buff=0.1)

        self.play(
            LaggedStart(
                Write(ax_title),
                GrowFromEdge(ax_title_underline, LEFT),
                lag_ratio=0.2,
            ),
            run_time=2,
        )

        self.play(Create(ax), Write(labels))
        self.wait(0.5)
        self.play(Create(dot1))
        self.play(Write(dot1_legend))

        self.wait(0.7)

        self.play(Indicate(bottleneck, color=bottleneck.color))

        self.wait(0.4)

        self.play(Indicate(decoder, color=decoder.color))
        self.play(GrowArrow(arrow_out))
        self.play(FadeIn(output_data))

        self.wait(0.7)

        # Same thing with images
        self.next_section(skip_animations=False)

        self.play(FadeOut(input_data, output_data, arrow_in, arrow_out))
        table = SVGMobject(
            "images/table.svg",
            opacity=1,
            fill_opacity=1,
            stroke_color=WHITE,
            fill_color=WHITE,
        ).scale(0.9)

        table.next_to(autoencoder, direction=LEFT, buff=1.5)
        arrow_in = Arrow(table.get_right(), autoencoder.get_left(), buff=0.2)

        dot2 = Dot(ax.c2p(0.7, 0.5), color=BLUE)
        dot2_legend = (
            Tex("Encoded table", color=BLUE)
            .scale(0.6)
            .next_to(dot2, direction=DOWN, buff=0.2)
        )

        self.play(FadeIn(table))
        self.play(GrowArrow(arrow_in))
        self.wait(0.5)
        self.play(Create(dot2))
        self.play(Write(dot2_legend))

        self.wait(0.4)

        self.play(FadeOut(table, arrow_in))
        input_img = ImageMobject("images/rabbit.png").scale(0.3)
        output_img = input_img.copy()
        dot3 = Dot(ax.c2p(0.2, 0.3), color=GREEN)
        dot3_legend = (
            Tex("Encoded rabbit", color=GREEN)
            .scale(0.6)
            .next_to(dot3, direction=DOWN, buff=0.2)
        )
        input_img.next_to(autoencoder, direction=LEFT, buff=1.5)
        output_img.next_to(autoencoder, direction=RIGHT, buff=1.5)
        arrow_in = Arrow(input_img.get_right(), autoencoder.get_left(), buff=0.2)
        arrow_out = Arrow(autoencoder.get_right(), output_img.get_left(), buff=0.2)

        self.play(FadeIn(input_img))
        self.play(GrowArrow(arrow_in))

        self.play(Create(dot3))
        self.play(Write(dot3_legend))
        self.play(GrowArrow(arrow_out))
        self.play(FadeIn(output_img))

        # Transform image into 2D points
        self.next_section(skip_animations=False)

        self.wait(0.5)

        # FadeOut previous objects
        self.play(
            FadeOut(
                encoder,
                bottleneck,
                decoder,
                input_img,
                output_img,
                dot1,
                dot1_legend,
                ax,
                dot2,
                dot2_legend,
                dot3,
                dot3_legend,
                labels,
                arrow_in,
                arrow_out,
                encoder_txt,
                bottleneck_txt,
                decoder_txt,
                ax_title,
                ax_title_underline,
            )
        )

        self.wait(2)


if __name__ == "__main__":
    scene = Scene2_1()
    scene.render()
