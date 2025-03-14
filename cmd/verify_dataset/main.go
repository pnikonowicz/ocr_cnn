package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path"
)

func log(message string) {
	fmt.Println(message)
}

func printAndTerminate(message string) {
	log(message)
	os.Exit(1)
}

func verifyTwoColors(img image.Image) {
	bounds := img.Bounds()
	expectedColorBlack := color.RGBA{0, 0, 0, 255}
	expectedColorWhite := color.RGBA{255, 255, 255, 255}

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r32,g32,b32,a32 := img.At(x, y).RGBA()
			r8, g8, b8, a8 := uint8(r32>>8), uint8(g32>>8), uint8(b32>>8), uint8(a32>>8)
			pixelColor := color.RGBA {
				R: r8,
				G: g8,
				B: b8,
				A: a8,
			}

			if pixelColor != expectedColorBlack &&
			   pixelColor != expectedColorWhite {
				printAndTerminate(fmt.Sprintf("got unexpected color: %x", pixelColor))
			}
		}
	}
}

func main() {
	wd, _ := os.Getwd()
	dataset_dir := path.Join(wd, "translated_dataset")

	expected_width := 64
	expected_height := 64

	for number := range []int{0,1,2,3,4,5,6,7,8,9} {

		dir := path.Join(dataset_dir, string(rune('0' + number)))
		dir_iterator, err := os.ReadDir(dir)

		log(fmt.Sprintf("checking dir: %s", dir))

		if(err != nil) {
			printAndTerminate("could not read dir")
		}

		for _, file_entry := range dir_iterator {
			file_name := path.Join(dir, file_entry.Name())
			file_contents, err := os.ReadFile(file_name)
			if err != nil {
				printAndTerminate(fmt.Sprintf("could not read file: %s", file_name))
			}

			img, err := png.Decode(bytes.NewReader(file_contents))
			if err != nil {
				printAndTerminate(fmt.Sprintf("could not read png: %s", file_name))
			}

			verifyImageResolution(img, expected_width, expected_height)
			verifyTwoColors(img)
		}
	}

	log("all images have same resolution")
	log("all images only have two colors")
}

func verifyImageResolution(img image.Image, expected_width int, expected_height int) {
	bounds := img.Bounds()
	width := bounds.Max.X - bounds.Min.X
	height := bounds.Max.Y - bounds.Min.Y

	if width != expected_width || height != expected_height {
		printAndTerminate(fmt.Sprintf("bad image: w: %d h: %d", width, height))
	}
}
