package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"ocr_cnn/pkg/common"
	"os"
	"path"
)

func translatePixels(original_img image.Image) image.Image {
	bounds := original_img.Bounds()

	// Create a new image with the same dimensions
	newImg := image.NewRGBA(bounds)

	// Define the threshold for "almost black"
	// Pixels with average RGB value below this threshold will be black
	// Adjust this value as needed (0-255)
	threshold := 30

	// Process each pixel
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixelColor := original_img.At(x, y)
			r, g, b, _ := pixelColor.RGBA()

			// Convert to 8-bit color values
			r, g, b = r>>8, g>>8, b>>8

			// Calculate average RGB value (ignoring alpha)
			avgColor := (r + g + b) / 3

			// Set new pixel color based on threshold
			if avgColor < uint32(threshold) {
				// Set to black
				newImg.Set(x, y, color.RGBA{0, 0, 0, 255})
			} else {
				// Set to white
				newImg.Set(x, y, color.RGBA{255, 255, 255, 255})
			}
		}
	}

	return newImg
}

func saveFile(dest_file_name string, translated_image image.Image) {
	// Create the output file
	outputFile, err := os.Create(dest_file_name)
	if err != nil {
		common.PrintAndTerminate(fmt.Sprintf("could not create output file: %s %s", dest_file_name, err.Error()))
	}
	defer outputFile.Close()

	// Encode and save the new image
	if err := png.Encode(outputFile, translated_image); err != nil {
		common.PrintAndTerminate(fmt.Sprintf("could not encode PNG: %s", dest_file_name))
	}
}

func main() {
	wd, _ := os.Getwd()
	dataset_source_dir := path.Join(wd, "dataset")
	dataset_dest_dir := path.Join(wd, "translated_dataset")

	for number := range []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9} {
		number_dir := string(rune('0' + number))
		source_dir := path.Join(dataset_source_dir, number_dir)
		dest_dir := path.Join(dataset_dest_dir, number_dir)
		dir_iterator, err := os.ReadDir(source_dir)

		if err := os.MkdirAll(dest_dir, 0755); err != nil {
			common.PrintAndTerminate(fmt.Sprintf("could not create output directory: %s", dest_dir))
		}

		if err != nil {
			common.PrintAndTerminate("could not read dir")
		}

		for _, file_entry := range dir_iterator {
			source_file_name := path.Join(source_dir, file_entry.Name())
			dest_file_name := path.Join(dest_dir, file_entry.Name())
			file_contents, err := os.ReadFile(source_file_name)
			if err != nil {
				common.PrintAndTerminate(fmt.Sprintf("could not read file: %s", source_file_name))
			}

			img, err := png.Decode(bytes.NewReader(file_contents))
			if err != nil {
				common.PrintAndTerminate(fmt.Sprintf("could not read png: %s", source_file_name))
			}

			var _ string = dest_file_name
			translated_image := translatePixels(img)

			saveFile(dest_file_name, translated_image)
		}
	}
}
