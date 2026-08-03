#[derive(Clone)]
pub struct State {
    pub node_ids_to_values: std::collections::HashMap<crate::model::node::Id, EValue>,
}

#[derive(Clone)]
pub enum EValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    String(String),
    Char(char),
    Undefined(String),
}

impl State {
    fn empty() -> Self {
        Self {
            node_ids_to_values: std::collections::HashMap::new(),
        }
    }

    pub fn new(
        ast: &crate::model::ast::Ast,
        user_vardecl_values: &std::collections::HashMap<crate::model::node::Id, EValue>,
        function_declarations: &std::collections::HashMap<
            crate::model::function_declaration::FunctionDeclarationId,
            crate::model::function_declaration::FunctionDeclaration,
        >,
    ) -> Result<Self, Vec<String>> {
        ast.get_sink_node_id()
            .ok_or(vec!["sink node id not present".to_owned()])
            .and_then(|sink_node_id| {
                ast.nodes
                    .get(&sink_node_id)
                    .cloned()
                    .ok_or(vec![format!("sink node {} not found", sink_node_id)])
                    .and_then(|sink_node| {
                        Self::empty().eval_next_step(
                            ast,
                            user_vardecl_values,
                            (sink_node_id, sink_node),
                            function_declarations,
                        )
                    })
            })
    }

    pub fn eval_next_step(
        &self,
        ast: &crate::model::ast::Ast,
        user_vardecl_values: &std::collections::HashMap<crate::model::node::Id, EValue>,
        (visitor_node_id, visitor_node): (crate::model::node::Id, crate::model::node::ENode),
        function_declarations: &std::collections::HashMap<
            crate::model::function_declaration::FunctionDeclarationId,
            crate::model::function_declaration::FunctionDeclaration,
        >,
    ) -> Result<Self, Vec<String>> {
        if self.node_ids_to_values.contains_key(&visitor_node_id) {
            Ok(self.clone())
        } else {
            let input_anchor_ids_to_node_ids =
                ast.get_connected_nodes_to_node_input_anchors(&visitor_node_id);
            let input_anchor_ids_to_values = input_anchor_ids_to_node_ids
                .iter()
                .filter_map(|(anchor_id, node_id)| {
                    self.node_ids_to_values
                        .get(node_id)
                        .map(|value| (anchor_id.clone(), value.clone()))
                })
                .collect::<std::collections::HashMap<_, _>>();
            if input_anchor_ids_to_node_ids.len() == input_anchor_ids_to_values.len() {
                Self::eval_value_for_node(
                    &visitor_node_id,
                    visitor_node,
                    input_anchor_ids_to_values,
                    user_vardecl_values,
                    ast,
                    function_declarations,
                )
                .map(|value| Self {
                    node_ids_to_values: self
                        .node_ids_to_values
                        .clone()
                        .into_iter()
                        .chain(vec![(visitor_node_id, value)])
                        .collect(),
                })
                .map_err(|error| vec![error])
            } else {
                let next_step_sub_evals = input_anchor_ids_to_node_ids
                    .into_iter()
                    .map(|(_, node_id)| match ast.nodes.get(&node_id).cloned() {
                        Some(node) => self.eval_next_step(
                            ast,
                            user_vardecl_values,
                            (node_id, node),
                            function_declarations,
                        ),
                        None => Err(vec![format!("node {} not found", node_id)]),
                    })
                    .collect::<Vec<Result<Self, Vec<String>>>>();
                let errors = next_step_sub_evals
                    .iter()
                    .filter_map(|result| result.as_ref().err())
                    .flatten()
                    .cloned()
                    .collect::<Vec<String>>();
                if errors.is_empty() {
                    Ok(Self {
                        node_ids_to_values: self
                            .node_ids_to_values
                            .clone()
                            .into_iter()
                            .chain(
                                next_step_sub_evals
                                    .into_iter()
                                    .filter_map(Result::ok)
                                    .flat_map(|state| state.node_ids_to_values),
                            )
                            .collect(),
                    })
                } else {
                    Err(errors)
                }
            }
        }
    }

    pub fn eval_value_for_node(
        node_id: &crate::model::node::Id,
        node: crate::model::node::ENode,
        input_anchor_ids_to_values: std::collections::HashMap<crate::model::anchor::Id, EValue>,
        user_vardecl_values: &std::collections::HashMap<crate::model::node::Id, EValue>,
        ast: &crate::model::ast::Ast,
        function_declarations: &std::collections::HashMap<
            crate::model::function_declaration::FunctionDeclarationId,
            crate::model::function_declaration::FunctionDeclaration,
        >,
    ) -> Result<EValue, String> {
        match node {
            crate::model::node::ENode::Program {} => {
                Err("cannot get value of a program node".to_string())
            }
            crate::model::node::ENode::Sink { input_anchor } => input_anchor_ids_to_values
                .get(&input_anchor)
                .cloned()
                .ok_or("no value found for input node for sink".to_string()),
            crate::model::node::ENode::FunctionCall {
                function_declaration_id,
                input_anchors,
                ..
            } => {
                let mut sorted_input_anchors = input_anchors.clone();
                sorted_input_anchors.sort_by_key(|anchor_id| match ast.anchors.get(anchor_id) {
                    Some(crate::model::anchor::EAnchor::Input(input_anchor)) => {
                        input_anchor.order_num
                    }
                    _ => usize::MAX,
                });
                let arguments = sorted_input_anchors
                    .iter()
                    .map(|anchor_id| {
                        input_anchor_ids_to_values
                            .get(anchor_id)
                            .cloned()
                            .ok_or(format!("no value found for input anchor {:?}", anchor_id))
                    })
                    .collect::<Result<Vec<EValue>, String>>()?;
                function_declarations
                    .get(&function_declaration_id)
                    .ok_or(format!(
                        "function declaration with id {} not found",
                        function_declaration_id.0
                    ))
                    .and_then(|function_declaration| {
                        Self::eval_value_for_function_call(function_declaration, arguments)
                    })
            }
            crate::model::node::ENode::ConstDecl { r#type, .. } => {
                Self::eval_value_for_type(r#type)
            }
            crate::model::node::ENode::TypeCast {
                r#type,
                input_anchor,
                ..
            } => input_anchor_ids_to_values
                .get(&input_anchor)
                .ok_or("no value found for input node for type cast".to_string())
                .map(|value| Self::eval_value_for_type_cast(value.clone(), r#type)),
            crate::model::node::ENode::VarDecl { name, .. } => user_vardecl_values
                .get(node_id)
                .cloned()
                .ok_or(format!("no value for var-decl provided: {}", name)),
            crate::model::node::ENode::Match { .. } => {
                Err("match eval not implemented yet".to_string())
            }
            crate::model::node::ENode::Pattern { .. } => {
                Err("pattern eval not implemented yet".to_string())
            }
        }
    }

    fn eval_value_for_type(r#type: crate::model::r#type::EType) -> Result<EValue, String> {
        match r#type {
            crate::model::r#type::EType::Bool { value } => value
                .ok_or("bool type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<bool>()
                        .map(EValue::Bool)
                        .map_err(|_| format!("could not parse \"{}\" as bool", v))
                }),
            crate::model::r#type::EType::Int { value } => value
                .ok_or("int type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<i32>()
                        .map(EValue::Int)
                        .map_err(|_| format!("could not parse \"{}\" as int", v))
                }),
            crate::model::r#type::EType::Float { value } => value
                .ok_or("float type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<f32>()
                        .map(EValue::Float)
                        .map_err(|_| format!("could not parse \"{}\" as float", v))
                }),
            crate::model::r#type::EType::String { value } => value
                .map(|v| EValue::String(v))
                .ok_or("string type did not have a specific value!".to_string()),
            crate::model::r#type::EType::Char { value } => value
                .ok_or("char type did not have a specific value!".to_string())
                .and_then(|v| {
                    v.parse::<char>()
                        .map(EValue::Char)
                        .map_err(|_| format!("could not parse \"{}\" as char", v))
                }),
            crate::model::r#type::EType::Any => {
                Err("any-type cannot provide a specific value!".to_string())
            }
            crate::model::r#type::EType::Undefined { message } => message
                .map(|v| EValue::Undefined(v))
                .ok_or("undefined type did not have a specific value!".to_string()),
        }
    }

    fn eval_value_for_function_call(
        function_declaration: &crate::model::function_declaration::FunctionDeclaration,
        arguments: Vec<EValue>,
    ) -> Result<EValue, String> {
        match (function_declaration.name.as_str(), arguments.as_slice()) {
            ("+", [EValue::Int(a), EValue::Int(b)]) => Ok(EValue::Int(a.wrapping_add(*b))),
            ("/", [EValue::Int(a), EValue::Int(b)]) => {
                if *b == 0 {
                    Ok(EValue::Undefined("division by zero".to_string()))
                } else {
                    Ok(EValue::Float(*a as f32 / *b as f32))
                }
            }
            ("charAt", [EValue::String(s), EValue::Int(i)]) => {
                match usize::try_from(*i).ok().and_then(|i| s.chars().nth(i)) {
                    Some(c) => Ok(EValue::Char(c)),
                    None => Ok(EValue::Undefined(format!(
                        "charAt: index {} out of bounds for string of length {}",
                        i,
                        s.chars().count()
                    ))),
                }
            }
            ("*(-1)", [EValue::Int(a)]) => Ok(EValue::Int(a.wrapping_neg())),
            ("substr", [EValue::String(s), EValue::Int(begin), EValue::Int(length)]) => {
                let chars = s.chars().collect::<Vec<char>>();
                let start = usize::try_from(*begin).unwrap_or(0).min(chars.len());
                let end = start
                    .saturating_add(usize::try_from(*length).unwrap_or(0))
                    .min(chars.len());
                Ok(EValue::String(chars[start..end].iter().collect()))
            }
            (name @ ("+" | "/" | "charAt" | "*(-1)" | "substr"), args) => Err(format!(
                "function {} called with unexpected argument types ({})",
                name,
                args.iter()
                    .map(EValue::type_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            (name, _) => Err(format!("unknown function declaration {}", name)),
        }
    }

    fn eval_value_for_type_cast(
        input_value: EValue,
        target_type: crate::model::r#type::EType,
    ) -> EValue {
        match &target_type {
            crate::model::r#type::EType::Bool { value: Some(_) }
            | crate::model::r#type::EType::Int { value: Some(_) }
            | crate::model::r#type::EType::Float { value: Some(_) }
            | crate::model::r#type::EType::String { value: Some(_) }
            | crate::model::r#type::EType::Char { value: Some(_) } => {
                return Self::eval_value_for_type(target_type).unwrap_or_else(EValue::Undefined);
            }
            _ => {}
        }

        if let EValue::Undefined(message) = input_value {
            return EValue::Undefined(message);
        }

        let input_type = input_value.type_name();
        match target_type {
            crate::model::r#type::EType::Bool { .. } => match input_value {
                EValue::Bool(b) => EValue::Bool(b),
                EValue::Int(i) => EValue::Bool(i != 0),
                EValue::Float(f) => EValue::Bool(f != 0.0),
                EValue::String(s) => s.parse::<bool>().map(EValue::Bool).unwrap_or_else(|_| {
                    EValue::Undefined(format!("cannot cast string \"{}\" to bool", s))
                }),
                _ => EValue::Undefined(format!("cannot cast {} to bool", input_type)),
            },
            crate::model::r#type::EType::Int { .. } => match input_value {
                EValue::Bool(b) => EValue::Int(if b { 1 } else { 0 }),
                EValue::Int(i) => EValue::Int(i),
                EValue::Float(f) => EValue::Int(f as i32),
                EValue::Char(c) => EValue::Int(c as i32),
                EValue::String(s) => s.parse::<i32>().map(EValue::Int).unwrap_or_else(|_| {
                    EValue::Undefined(format!("cannot cast string \"{}\" to int", s))
                }),
                _ => EValue::Undefined(format!("cannot cast {} to int", input_type)),
            },
            crate::model::r#type::EType::Float { .. } => match input_value {
                EValue::Bool(b) => EValue::Float(if b { 1.0 } else { 0.0 }),
                EValue::Int(i) => EValue::Float(i as f32),
                EValue::Float(f) => EValue::Float(f),
                EValue::String(s) => s.parse::<f32>().map(EValue::Float).unwrap_or_else(|_| {
                    EValue::Undefined(format!("cannot cast string \"{}\" to float", s))
                }),
                _ => EValue::Undefined(format!("cannot cast {} to float", input_type)),
            },
            crate::model::r#type::EType::String { .. } => EValue::String(match input_value {
                EValue::Bool(b) => b.to_string(),
                EValue::Int(i) => i.to_string(),
                EValue::Float(f) => f.to_string(),
                EValue::String(s) => s,
                EValue::Char(c) => c.to_string(),
                EValue::Undefined(message) => message,
            }),
            crate::model::r#type::EType::Char { .. } => match input_value {
                EValue::Char(c) => EValue::Char(c),
                EValue::Int(i) => u32::try_from(i)
                    .ok()
                    .and_then(char::from_u32)
                    .map(EValue::Char)
                    .unwrap_or_else(|| EValue::Undefined(format!("cannot cast int {} to char", i))),
                EValue::String(s) => s.parse::<char>().map(EValue::Char).unwrap_or_else(|_| {
                    EValue::Undefined(format!("cannot cast string \"{}\" to char", s))
                }),
                _ => EValue::Undefined(format!("cannot cast {} to char", input_type)),
            },
            crate::model::r#type::EType::Any => input_value,
            crate::model::r#type::EType::Undefined { message } => {
                EValue::Undefined(message.unwrap_or_else(|| "undefined".to_string()))
            }
        }
    }

    /// True once the sink node carries a value, i.e. evaluation has reached the
    /// root and stepping further would not add anything. Used to grey out the
    /// `Next` button.
    pub fn is_evaluated(&self, ast: &crate::model::ast::Ast) -> bool {
        ast.get_sink_node_id()
            .is_some_and(|sink_id| self.node_ids_to_values.contains_key(&sink_id))
    }
}

impl std::fmt::Display for EValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EValue::Bool(value) => write!(f, "{}", value),
            EValue::Int(value) => write!(f, "{}", value),
            EValue::Float(value) => write!(f, "{}", value),
            EValue::String(value) => write!(f, "{}", value),
            EValue::Char(value) => write!(f, "{}", value),
            EValue::Undefined(_) => write!(f, "undefined"),
        }
    }
}

impl EValue {
    /// Parse a user-typed VarDecl input string into a value of the declared
    /// type. VarDecls carry their value in `user_vardecl_values` (their type
    /// literal is `None` at declaration time), so this is the path from the
    /// prompt modal into evaluation.
    pub fn parse(target: &crate::model::r#type::EType, raw: &str) -> Result<EValue, String> {
        match target {
            crate::model::r#type::EType::Bool { .. } => raw
                .parse::<bool>()
                .map(EValue::Bool)
                .map_err(|_| format!("could not parse \"{}\" as bool", raw)),
            crate::model::r#type::EType::Int { .. } => raw
                .parse::<i32>()
                .map(EValue::Int)
                .map_err(|_| format!("could not parse \"{}\" as int", raw)),
            crate::model::r#type::EType::Float { .. } => raw
                .parse::<f32>()
                .map(EValue::Float)
                .map_err(|_| format!("could not parse \"{}\" as float", raw)),
            crate::model::r#type::EType::String { .. } => Ok(EValue::String(raw.to_string())),
            crate::model::r#type::EType::Char { .. } => raw
                .parse::<char>()
                .map(EValue::Char)
                .map_err(|_| format!("could not parse \"{}\" as char", raw)),
            crate::model::r#type::EType::Any => {
                Err("any-typed var-decl needs a concrete type".to_string())
            }
            crate::model::r#type::EType::Undefined { .. } => Ok(EValue::Undefined(raw.to_string())),
        }
    }

    fn type_name(&self) -> &'static str {
        match self {
            EValue::Bool(_) => "bool",
            EValue::Int(_) => "int",
            EValue::Float(_) => "float",
            EValue::String(_) => "string",
            EValue::Char(_) => "char",
            EValue::Undefined(_) => "undefined",
        }
    }
}
