//! Practice sentences (`lipflow/practice.py`, text part). Clip storage lives elsewhere.
//!
//! Training holds [`N_HELD_OUT`] recorded clips out and only keeps the face model if it reads
//! those better than the stock model.

use std::io;
use std::path::Path;
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;

pub const N_SENTENCES: usize = 24;

/// Harvard sentences (IEEE 1969 "Recommended Practice for Speech Quality Measurements", lists 1-6):
/// short, phonetically balanced sentences, so practice covers every lip shape evenly. Used when
/// there's no Wispr Flow history to practice your own sentences from.
pub const HARVARD: [&str; 60] = [
    "The birch canoe slid on the smooth planks",
    "Glue the sheet to the dark blue background",
    "It's easy to tell the depth of a well",
    "These days a chicken leg is a rare dish",
    "Rice is often served in round bowls",
    "The juice of lemons makes fine punch",
    "The box was thrown beside the parked truck",
    "The hogs were fed chopped corn and garbage",
    "Four hours of steady work faced us",
    "A large size in stockings is hard to sell",
    "The boy was there when the sun rose",
    "A rod is used to catch pink salmon",
    "The source of the huge river is the clear spring",
    "Kick the ball straight and follow through",
    "Help the woman get back to her feet",
    "A pot of tea helps to pass the evening",
    "Smoky fires lack flame and heat",
    "The soft cushion broke the man's fall",
    "The salt breeze came across from the sea",
    "The girl at the booth sold fifty bonds",
    "The small pup gnawed a hole in the sock",
    "The fish twisted and turned on the bent hook",
    "Press the pants and sew a button on the vest",
    "The swan dive was far short of perfect",
    "The beauty of the view stunned the young boy",
    "Two blue fish swam in the tank",
    "Her purse was full of useless trash",
    "The colt reared and threw the tall rider",
    "It snowed rained and hailed the same morning",
    "Read verse out loud for pleasure",
    "Hoist the load to your left shoulder",
    "Take the winding path to reach the lake",
    "Note closely the size of the gas tank",
    "Wipe the grease off his dirty face",
    "Mend the coat before you go out",
    "The wrist was badly strained and hung limp",
    "The stray cat gave birth to kittens",
    "The young girl gave no clear response",
    "The meal was cooked before the bell rang",
    "What joy there is in living",
    "A king ruled the state in the early days",
    "The ship was torn apart on the sharp reef",
    "Sickness kept him home the third week",
    "The wide road shimmered in the hot sun",
    "The lazy cow lay in the cool grass",
    "Lift the square stone over the fence",
    "The rope will bind the seven books at once",
    "Hop over the fence and plunge in",
    "The friendly gang left the drug store",
    "Mesh wire keeps chicks inside",
    "The frosty air passed through the coat",
    "The crooked maze failed to fool the mouse",
    "Adding fast leads to wrong sums",
    "The show was a flop from the very start",
    "A saw is a tool used for making boards",
    "The wagon moved on well oiled wheels",
    "March the soldiers past the next hill",
    "A cup of sugar makes sweet fudge",
    "Place a rosebush near the porch steps",
    "Both lost their lives in the raging storm",
];

/// SplitMix64, seeded from the clock: enough to vary practice order between sessions.
struct Rng(u64);

impl Rng {
    fn from_clock() -> Self {
        Self(SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Fisher–Yates.
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }

    /// `k` distinct elements in random order (`random.sample`).
    fn sample<T: Clone>(&mut self, v: &[T], k: usize) -> Vec<T> {
        let mut pool = v.to_vec();
        let k = k.min(pool.len());
        for i in 0..k {
            let j = i + self.below(pool.len() - i);
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool
    }
}

static SENTENCE_GAP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[.!?](\s+)").expect("static regex"));
static NOT_SAYABLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\d|http|@|/").expect("static regex"));

/// `re.split(r"(?<=[.!?])\s+", line)`: split at whitespace that follows end punctuation.
fn sentences(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for c in SENTENCE_GAP.captures_iter(line) {
        if let Some(gap) = c.get(1) {
            out.push(&line[start..gap.start()]);
            start = gap.end();
        }
    }
    out.push(&line[start..]);
    out
}

/// Everyday Russian sentences for the practice round: work and home, varied lip shapes.
pub const RUSSIAN: [&str; 60] = [
    "Привет, как у тебя дела сегодня",
    "Давай созвонимся завтра утром после девяти",
    "Я отправил тебе файл на почту",
    "Мне нужно купить хлеб и молоко",
    "Встреча переносится на пятницу в три часа",
    "Посмотри, пожалуйста, этот документ до вечера",
    "Я немного опоздаю, начинайте без меня",
    "Спасибо большое за помощь с проектом",
    "Можешь прислать мне ссылку на презентацию",
    "Мы обсудим это на следующей неделе",
    "Не забудь взять с собой зонтик",
    "Позвони мне, когда освободишься",
    "Я уже выхожу из дома",
    "Сегодня очень тёплая и солнечная погода",
    "Давай закажем пиццу на ужин",
    "Отличная идея, давай так и сделаем",
    "Нужно перенести релиз на понедельник",
    "Я проверю код и напишу тебе",
    "Пожалуйста, обнови таблицу с бюджетом",
    "Кто будет вести завтрашнюю встречу",
    "Мы успеваем закончить работу к сроку",
    "Купи по дороге фрукты и овощи",
    "Я забронировал столик на семь вечера",
    "У меня сломался ноутбук, работаю с телефона",
    "Пришли мне, пожалуйста, номер его телефона",
    "Поздравляю тебя с днём рождения",
    "Буду рад видеть вас в гостях",
    "Отправь отчёт руководителю до обеда",
    "Давай встретимся у входа в метро",
    "Я посмотрел фильм, он мне понравился",
    "Нам нужно больше времени на тестирование",
    "Сколько стоит доставка до офиса",
    "Предлагаю выпить кофе после работы",
    "Мама просила позвонить ей вечером",
    "Пожалуйста, не забудь выключить свет",
    "Поезд отправляется в восемь пятнадцать",
    "Я напомню об этом завтра утром",
    "Какие у тебя планы на выходные",
    "Давай пройдёмся пешком до парка",
    "Мне понравился твой новый дизайн",
    "Нужно позвонить врачу и записаться на приём",
    "Возьми, пожалуйста, ключи от машины",
    "Я перешлю тебе письмо от клиента",
    "Добавь меня в общий чат проекта",
    "Мы переезжаем в новый офис в мае",
    "Подожди минутку, я сейчас вернусь",
    "Положи документы в синюю папку",
    "Вода в бассейне сегодня холодная",
    "Пора обновить приложение на телефоне",
    "Я буду дома примерно через час",
    "Приятного аппетита и хорошего вечера",
    "Напиши мне, если появятся вопросы",
    "Завтра обещают сильный дождь и ветер",
    "Пусть каждый подготовит короткий доклад",
    "Мы выпустили новую версию программы",
    "Проверь, пожалуйста, мою орфографию в тексте",
    "Сегодня я работаю из дома",
    "Будь добр, перезвони мне попозже",
    "Мне нравится гулять вечером по набережной",
    "Обязательно передай привет своим родителям",
];

/// Half your own everyday sentences (from imported phrases at `phrases`: 5–12 words, no digits,
/// links, @ or /) for your real vocabulary, the rest Harvard sentences for even coverage of lip
/// shapes; all Harvard if there's no history. Shuffled together.
/// Only your sentences in the dictation language's script count, and Russian uses the Russian
/// list instead of the Harvard sentences.
pub fn practice_sentences_in(n: usize, phrases: &Path, lang: crate::text::Lang) -> io::Result<Vec<String>> {
    let ru = lang == crate::text::Lang::Ru;
    let in_script = |s: &str| s.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c)) == ru;
    let mut rng = Rng::from_clock();
    let mut mine: Vec<String> = Vec::new();
    match std::fs::read_to_string(phrases) {
        Ok(text) => {
            for line in text.lines() {
                for s in sentences(line.trim()) {
                    let w = s.split_whitespace().count();
                    if (5..=12).contains(&w) && !NOT_SAYABLE.is_match(s) && in_script(s) {
                        mine.push(s.trim_end_matches(['.', '!', '?', ',']).to_string());
                    }
                }
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    rng.shuffle(&mut mine);
    let mut own: Vec<String> = Vec::new();
    for s in mine {
        if own.len() == n / 2 {
            break;
        }
        if !own.contains(&s) {
            own.push(s);
        }
    }
    let harvard = rng.sample(if ru { &RUSSIAN } else { &HARVARD }, n - own.len());
    let mut out = own;
    out.extend(harvard.into_iter().map(str::to_string));
    rng.shuffle(&mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    fn temp_home(tag: &str) -> PathBuf {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("lipflow-{tag}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    // Ported from lipflow/tests/test_cleanup.py
    #[test]
    fn russian_practice_uses_russian_sentences() {
        let home = std::env::temp_dir().join(format!("lipflow-ru-practice-{}", std::process::id()));
        let s = practice_sentences_in(24, &home.join("none.txt"), crate::text::Lang::Ru).unwrap();
        assert_eq!(s.len(), 24);
        assert!(s.iter().all(|x| RUSSIAN.contains(&x.as_str())));
    }

    #[test]
    fn practice_sentences_fall_back_to_harvard() {
        let home = temp_home("practice-none");
        let s = practice_sentences_in(24, &home.join("none.txt"), crate::text::Lang::En).unwrap();
        assert!(s.len() == 24 && s.iter().collect::<HashSet<_>>().len() == 24 && s.iter().all(|x| HARVARD.contains(&x.as_str())));
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn practice_mixes_own_and_harvard() {
        let home = temp_home("practice-mix");
        let f = home.join("phrases.txt");
        let lines: Vec<String> = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
            .split(' ')
            .map(|w| format!("This is my own sentence number {w} for testing"))
            .collect();
        fs::write(&f, lines.join("\n")).unwrap();
        let s = practice_sentences_in(24, &f, crate::text::Lang::En).unwrap();
        assert_eq!(s.iter().filter(|x| HARVARD.contains(&x.as_str())).count(), 12);
        assert_eq!(s.iter().collect::<HashSet<_>>().len(), 24);
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn splits_after_end_punctuation_only() {
        assert_eq!(sentences("Hi there. How are you?  Fine e.g.x ok"), ["Hi there.", "How are you?", "Fine e.g.x ok"]);
    }
}
