use json;
use rayon;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use std::fs::{self, copy, DirEntry, File};
use std::io::{BufRead, BufReader, Error, Read, Write};
use std::path::{Display, Path, PathBuf};
use wav;

fn calculate_wav_length(wav: &PathBuf) -> f64 {
    // let buf_input_file = BufReader::new(File::open(wav).unwrap());

    let mut input_file: File = match File::open(wav) {
        Ok(val) => val,
        Err(err) => {
            println!("{:?}", err);
            return 0.0;
        }
    };
    let file_size: f64 = input_file.metadata().unwrap().len() as f64;

    let (header, _data) = match wav::read(&mut input_file) {
        Ok(f) => f,
        Err(_err) => return 0.0,
    };
    let bytes_per_second: f64 = header.bytes_per_second as f64;
    let seconds: f64 = file_size / bytes_per_second;
    return seconds;
}

fn basic_dataset_validation(dataset: &PathBuf) -> Result<bool, Vec<String>> {
    let mut errs: Vec<String> = Vec::new();
    let dataset_to_display: &Display = &dataset.as_path().display();

    let str_path_to_train_files = format!("{}/{}", dataset_to_display, "list_train.txt");
    let str_path_to_val_files = format!("{}/{}", dataset_to_display, "list_val.txt");
    let str_path_to_wavs_files = format!("{}/{}", dataset_to_display, "wavs");

    let train_files = Path::new(&str_path_to_train_files);
    if !train_files.exists() {
        errs.push("list_train.txt not found".to_owned());
    }
    let val_files = Path::new(&str_path_to_val_files);
    if !val_files.exists() {
        errs.push("list_val.txt not found".to_owned());
    }
    let wavs_dir = Path::new(&str_path_to_wavs_files);
    if !wavs_dir.exists() {
        errs.push("wavs directory not found".to_owned());
    }
    if errs.is_empty() {
        return Ok(true);
    }
    return Err(errs);
}

struct LineData {
    path_for_copy: PathBuf,
    wav: String,
    text: String,
    audio_length: f64,
}
struct DatasetBasicData {
    id: usize,
    name: String,
    length: f64,
    waveglow: String,
    path_to_train_file: PathBuf,
    path_to_val_file: PathBuf,
    train_lines: Vec<LineData>,
    val_lines: Vec<LineData>,
}

fn main() {
    // Setting up basic directories
    let path_to_datasets: PathBuf = Path::new("datasets").to_path_buf();
    if !path_to_datasets.exists() {
        panic!("No folders with datasets. Quitting... Don't give up tho.");
    }

    let path_to_mixed_wavs: PathBuf = Path::new("mixed_wavs").to_path_buf();
    let path_to_mixed_lists: PathBuf = Path::new("mixed_lists").to_path_buf();
    let mut valid_datasets: Vec<PathBuf> = Vec::new();

    if !path_to_mixed_wavs.exists() {
        match fs::create_dir(path_to_mixed_wavs) {
            Ok(dir) => dir,
            Err(err) => panic!("Couldn't create mixed_wavs directory... Reason: {}", err),
        };
    };
    if !path_to_mixed_lists.exists() {
        match fs::create_dir(path_to_mixed_lists) {
            Ok(dir) => dir,
            Err(err) => panic!("Couldn't create mixed_lists directory... Reason: {}", err),
        };
    };

    // Defining file handlers for sets and model info
    let mut train_mixed_lists_path = PathBuf::new();
    train_mixed_lists_path.push("mixed_lists");
    train_mixed_lists_path.push("list_train.txt");
    let mut train_mixed_lists = File::create(train_mixed_lists_path).expect("Error occured while creating mixed_lists/list_train.txt");

    let mut val_mixed_lists_path = PathBuf::new();
    val_mixed_lists_path.push("mixed_lists");
    val_mixed_lists_path.push("list_val.txt");
    let mut val_mixed_lists = File::create(val_mixed_lists_path).expect("Error occured while creating mixed_lists/list_val.txt");

    let mut model_info_path = PathBuf::new();
    model_info_path.push("mixed_lists");
    model_info_path.push("model_info.json");
    let mut model_info_mixed_lists = File::create(model_info_path).expect("Error occured while creating mixed_lists/model_info.json");

    // Excluding invalid datasets (lacking of wavs or list_train.txt or list_val.txt)
    'valid_dataset_loop: for dataset in fs::read_dir(path_to_datasets).unwrap() {
        let entry = match dataset {
            Ok(_) => dataset.unwrap(),
            Err(err) => {
                println!("{}", err);
                continue;
            }
        };

        let mut path_to_entry = PathBuf::new();
        path_to_entry.push(Path::new("datasets"));
        path_to_entry.push(Path::new(&entry.file_name()));

        match basic_dataset_validation(&path_to_entry) {
            Ok(_) => true,
            Err(err) => {
                println!("Errors in dataset of {:?}", &entry.file_name());
                for el in err {
                    println!("\t{}", el);
                }
                println!("");
                continue 'valid_dataset_loop;
            }
        };
        valid_datasets.push(path_to_entry);
    }
    if valid_datasets.is_empty() {
        panic!("No valid dataset there!");
    }
    let mut datasets: Vec<DatasetBasicData> = Vec::new();
    'define_datasets_loop: for (index, dataset) in valid_datasets.into_iter().enumerate() {
        let mut train_lines: Vec<LineData> = Vec::new();
        let mut val_lines: Vec<LineData> = Vec::new();
        // Define paths
        let mut path_to_wavs = PathBuf::new();
        path_to_wavs.push(&dataset);
        path_to_wavs.push(Path::new("wavs"));

        let mut path_to_list_train = PathBuf::new();
        path_to_list_train.push(&dataset);
        path_to_list_train.push(Path::new("list_train.txt"));

        let mut path_to_list_val = PathBuf::new();
        path_to_list_val.push(&dataset);
        path_to_list_val.push(Path::new("list_val.txt"));
        // Define file handlers
        let list_train_file = File::open(path_to_list_train).unwrap();
        let list_val_file = File::open(path_to_list_val).unwrap();

        // Define BufReaders for current file handlers
        let list_train_buf = BufReader::new(&list_train_file);
        let list_val_buf = BufReader::new(&list_val_file);
        // Get data about lines in both lists
        for line in list_train_buf.lines() {
            let line_str = line.unwrap();
            let split_line: std::str::Split<&str> = line_str.split("|");

            let line_container: Vec<&str> = split_line.collect();
            let wav = line_container[0].to_string();
            let mut path_to_wav_buf = PathBuf::new();
            path_to_wav_buf.push(dataset.as_path());
            path_to_wav_buf.push(wav);

            let path_to_wav = Path::new(&path_to_wav_buf);
            if !path_to_wav_buf.exists() {
                println!(
                    "File {} does not exist for some reason.",
                    path_to_wav_buf.display()
                );
                continue 'define_datasets_loop;
            }
            let audio_length = calculate_wav_length(&path_to_wav_buf);

            let line_data = LineData {
                path_for_copy: path_to_wav.to_path_buf(),
                wav: line_container[0].to_string().replace("wavs/", ""),
                text: line_container[1].to_string(),
                audio_length,
            };
            train_lines.push(line_data);
        }
        for line in list_val_buf.lines() {
            let line_str = line.unwrap();
            let split_line: std::str::Split<&str> = line_str.split("|");

            let line_container: Vec<&str> = split_line.collect();
            let wav = line_container[0].to_string();
            let mut path_to_wav = PathBuf::new();
            path_to_wav.push(dataset.as_path());
            path_to_wav.push(wav);
            if !path_to_wav.exists() {
                println!(
                    "File '{}' does not exist for some reason.",
                    path_to_wav.display()
                );
                continue;
            }
            let audio_length = calculate_wav_length(&path_to_wav);
            if audio_length >= 10.0 {
                println!(
                    "{} has greater length than 10 (or it's equal). Just sayin'",
                    path_to_wav.display()
                );
            }
            let line_data = LineData {
                path_for_copy: path_to_wav,
                wav: line_container[0].to_string().replace("wavs/", ""),
                text: line_container[1].to_string(),
                audio_length,
            };
            val_lines.push(line_data);
        }
        // Calculates length in seconds
        let sum_length = match train_lines
            .iter()
            .chain(val_lines.iter())
            .map(|val| val.audio_length)
            .reduce(|acc, cur| acc + cur)
            .ok_or_else(|| 0.0)
        {
            Ok(val) => val,
            Err(_) => {
                println!(
                    "Double check dataset {:?} if train or val lists aren't blank or something.",
                    dataset
                );
                continue;
            }
        };
        if sum_length < 300.0 {
            println!(
                "Dataset {} is too short dataset. Discarding",
                &dataset.display()
            );
            continue;
        }
        const SECONDS_IN_MINUTE: f64 = 60.0;
        let length_in_hours = sum_length / SECONDS_IN_MINUTE;

        let mut path_to_list_train = PathBuf::new();
        path_to_list_train.push(&dataset);
        path_to_list_train.push(Path::new("list_train.txt"));

        let mut path_to_list_val = PathBuf::new();
        path_to_list_val.push(&dataset);
        path_to_list_val.push(Path::new("list_val.txt"));

        let dataset_data: DatasetBasicData = DatasetBasicData {
            id: index,
            name: dataset.file_name().unwrap().to_str().unwrap().to_string(),
            train_lines,
            val_lines,
            path_to_train_file: path_to_list_train,
            path_to_val_file: path_to_list_val,
            length: length_in_hours,
            waveglow: "Vatras".to_string(),
        };

        datasets.push(dataset_data);
    }
    let mut json_actors = json::array![];

    for (index, dataset) in datasets.iter().enumerate() {
        let actor_data = json::object! {
            "id": dataset.id,
            "name": dataset.name.as_str(),
            "waveglow": dataset.waveglow.as_str(),
            "length": dataset.length,
        };
        dataset.train_lines.iter().for_each(|x| {
            let flowtron_line = format!("wavs/{}_{}|{}|{}\n", index, x.wav, x.text, dataset.id);
            train_mixed_lists
                .write(flowtron_line.as_bytes())
                .expect("Failed to write data in mixed_lists/list_train.txt");
        });
        dataset.val_lines.iter().for_each(|x| {
            let flowtron_line = format!("wavs/{}_{}|{}|{}\n", index, x.wav, x.text, dataset.id);
            val_mixed_lists
                .write(flowtron_line.as_bytes())
                .expect("Failed to write data in mixed_lists/list_val.txt");
        });

        for entry in dataset.val_lines.iter().chain(dataset.train_lines.iter()) {
            let mut path_to_mixed_wavs: PathBuf = Path::new("mixed_wavs").to_path_buf();
            let wav_name = format!("{}_{}", &index, &entry.wav);
            path_to_mixed_wavs.push(wav_name);
            copy(entry.path_for_copy.as_path(), &path_to_mixed_wavs).expect(&format!(
                "Failed to copy file from {} to {}",
                entry.path_for_copy.display(),
                path_to_mixed_wavs.display()
            ));
        }
        json_actors[index] = actor_data;
    }

    let model_info_json = json::object! {
        "name": "",
        "n_speakers": json_actors.len(),
        "tacotron": "",
        "train_list": "",
        "actors": json_actors
    };

    model_info_json
        .write_pretty(&mut model_info_mixed_lists, 4)
        .expect("Couldn't write data into model_info");
    println!("Done.");
}
